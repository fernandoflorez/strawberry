from collections import OrderedDict
from collections.abc import Callable
from enum import Enum
from math import isfinite
from threading import Lock
from types import CodeType
from typing import Any, NamedTuple, cast
from weakref import ref

from graphql import (
    GraphQLBoolean,
    GraphQLError,
    GraphQLField,
    GraphQLFloat,
    GraphQLID,
    GraphQLInt,
    GraphQLObjectType,
    GraphQLOutputType,
    GraphQLResolveInfo,
    GraphQLString,
    OperationType,
    Undefined,
    is_abstract_type,
    is_leaf_type,
    is_list_type,
    is_non_null_type,
    is_object_type,
)
from graphql.execution.collect_fields import FieldDetailsList
from graphql.execution.executor import (
    collect_iterator_awaitables,
    invalid_return_type_error,
)
from graphql.execution.values import get_argument_values
from graphql.pyutils import AwaitableOrValue, Path, gather_with_cancel, is_iterable
from graphql.type.scalars import (
    GRAPHQL_MAX_INT,
    GRAPHQL_MIN_INT,
    serialize_boolean,
    serialize_float,
    serialize_id,
    serialize_int,
    serialize_string,
)

from strawberry.schema.schema import StrawberryGraphQLCoreExecutionContext
from strawberry.schema.schema_converter import (
    CustomGraphQLEnumType,
    GraphQLCoreConverter,
)
from strawberry.types.arguments import convert_arguments
from strawberry.types.field import StrawberryField
from strawberry.types.fields.resolver import StrawberryResolver
from strawberry.types.info import Info


def _nested_codes(functions: tuple[Any, ...], name: str) -> frozenset[CodeType]:
    found: set[CodeType] = set()
    stack = [function.__code__ for function in functions]
    while stack:
        code = stack.pop()
        for const in code.co_consts:
            if isinstance(const, CodeType):
                if const.co_name == name:
                    found.add(const)
                stack.append(const)
    return frozenset(found)


_RESOLVER_WRAPPER_CODES = _nested_codes(
    (GraphQLCoreConverter.from_resolver,), "_resolver"
) | _nested_codes((GraphQLCoreConverter.from_resolver,), "_async_resolver")
_BASIC_RESOLVER_CODES = _nested_codes(
    (GraphQLCoreConverter.from_resolver,), "_get_basic_result"
)
_IS_TYPE_OF_CODES = _nested_codes(
    (GraphQLCoreConverter.from_object, GraphQLCoreConverter._get_is_type_of),
    "is_type_of",
)
_TRUSTED_MODULES = frozenset(
    {
        "graphql.execution.executor",
        "graphql.execution.incremental.incremental_executor",
        "strawberry.schema.schema",
    }
)


def _code_of(function: Any) -> CodeType | None:
    return getattr(function, "__code__", None)


def _is_plannable_argument_type(type_: Any) -> bool:
    if is_non_null_type(type_):
        type_ = type_.of_type
    return type_ in _BUILTIN_SCALAR_TYPES or isinstance(type_, CustomGraphQLEnumType)


def _untouched(cls: type, name: str) -> bool:
    method = getattr(cls, name)
    if method is not getattr(FieldPlanExecutionContext, name):
        return False
    if name in vars(FieldPlanExecutionContext):
        return True
    return getattr(method, "__module__", None) in _TRUSTED_MODULES and not hasattr(
        method, "__wrapped__"
    )


_BYPASSED_METHODS = (
    "execute_field",
    "execute_fields",
    "build_resolve_info",
    "complete_value",
    "complete_awaitable_value",
    "complete_list_value",
    "complete_iterable_value",
    "complete_list_item_value",
    "complete_leaf_value",
    "complete_object_value",
    "complete_abstract_value",
    "collect_and_execute_subfields",
    "execute_collected_subfields",
    "get_stream_usage",
    "with_abort_signal",
)

_NON_NULL = 0
_LIST = 1
_LEAF = 2
_OBJECT = 3
_GENERIC = 4
_NON_NULL_LEAF = 5
_ABSTRACT = 6
_NATIVE_SEQUENCES = frozenset({list, tuple})
_ORIGINAL_RESOLVER: Any = object()
_SEEN_ONCE: Any = object()
_CONVERT_PER_CALL: Any = object()
_IMMUTABLE_ARGUMENT_TYPES = frozenset({str, int, float, bool, type(None)})

_BUILTIN_SCALARS = (
    (GraphQLString, serialize_string, str),
    (GraphQLID, serialize_id, str),
    (GraphQLBoolean, serialize_boolean, bool),
    (GraphQLInt, serialize_int, int),
    (GraphQLFloat, serialize_float, float),
)
_BUILTIN_SCALAR_TYPES = frozenset(scalar for scalar, _, _ in _BUILTIN_SCALARS)


_UNBOUNDED = 0
_INT_BOUNDS = 1
_FLOAT_BOUNDS = 2


def _leaf_fast(shape: Any) -> tuple[type, int, bool] | None:
    nullable = shape.kind == _LEAF
    if not nullable and shape.kind != _NON_NULL_LEAF:
        return None
    leaf = shape if nullable else shape.child
    exact_type = leaf.child
    if exact_type is None:
        return None
    bound = (
        _INT_BOUNDS
        if exact_type is int
        else _FLOAT_BOUNDS
        if exact_type is float
        else _UNBOUNDED
    )
    return exact_type, bound, nullable


_NOT_A_LIST = 0
_NON_NULL_ITEMS = 1
_NULLABLE_ITEMS = 2
_COERCE_LEAF = 3


def _all_exact(
    items: list[Any], exact_type: type, bound: int, items_nullable: bool
) -> bool:
    for item in items:
        if type(item) is exact_type:
            if bound == _UNBOUNDED or (
                GRAPHQL_MIN_INT <= item <= GRAPHQL_MAX_INT
                if bound == _INT_BOUNDS
                else isfinite(item)
            ):
                continue
            return False
        if item is None and items_nullable:
            continue
        return False
    return True


def _inline_getter(plan: Any) -> tuple[str, Any, int, bool, int] | None:
    if (
        plan is None
        or plan.resolver is not None
        or plan.default_resolver is not getattr
    ):
        return None
    if plan.leaf_fast is not None:
        return (plan.python_name, *plan.leaf_fast, _NOT_A_LIST)
    shape = plan.shape
    if shape.kind == _LEAF:
        return (plan.python_name, shape, _UNBOUNDED, True, _COERCE_LEAF)
    if shape.kind == _NON_NULL_LEAF:
        return (plan.python_name, shape.child, _UNBOUNDED, False, _COERCE_LEAF)
    nullable = shape.kind == _LIST
    if not nullable:
        if shape.kind != _NON_NULL:
            return None
        shape = shape.child
        if shape.kind != _LIST:
            return None
    item_fast = _leaf_fast(shape.child)
    if item_fast is None:
        return None
    exact_type, bound, items_nullable = item_fast
    return (
        plan.python_name,
        exact_type,
        bound,
        nullable,
        _NULLABLE_ITEMS if items_nullable else _NON_NULL_ITEMS,
    )


def _scalar_fast_path(leaf_type: Any) -> type | None:
    for scalar, serializer, exact_type in _BUILTIN_SCALARS:
        if leaf_type is scalar:
            return exact_type if leaf_type.coerce_output_value is serializer else None
    return None


class _Shape(NamedTuple):
    kind: int
    type_: Any
    child: Any
    is_type_of: Callable[[Any, Any], Any] | None
    enum_cache: tuple[type, dict[int, str]] | None = None


def _is_static_shape(shape: _Shape) -> bool:
    while shape.kind != _LEAF:
        if shape.kind in (_OBJECT, _GENERIC, _ABSTRACT):
            return False
        shape = shape.child
    return True


_ObjectEntry = tuple[str, FieldDetailsList, Any, tuple[str, Any, int, bool, int] | None]


class _FieldPlan(NamedTuple):
    field_def: GraphQLField
    field: StrawberryField | None
    field_name: str
    return_type: GraphQLOutputType
    shape: _Shape
    resolver: Callable[..., Any] | None
    pass_source: bool
    source_kwargs: tuple[str, ...]
    info_kwarg: str | None
    kwargs: list[dict[str, Any] | None]
    default_resolver: Callable[[Any, str], Any]
    python_name: str
    leaf_fast: tuple[type, int, bool] | None = None


_CLASS_ALLOWED: dict[type, bool] = {}
_PLAN_CACHE_SIZE = 256
_PLAN_CACHE_LOCK = Lock()
_FIELD_SCOPE_DECLARED: dict[type, bool] = {}


def _has_fragment_arguments(field_details_list: FieldDetailsList) -> bool:
    return any(
        details.fragment_variable_values is not None for details in field_details_list
    )


def _owner(cls: type, name: str) -> type | None:
    for klass in cls.__mro__:
        if name in vars(klass):
            return klass
    return None


def _declares_field_scope(cls: type) -> bool:
    declared = _FIELD_SCOPE_DECLARED.get(cls)
    if declared is None:
        owner = _owner(cls, "applies_to_field")
        declared = owner is not None and owner is _owner(cls, "resolve")
        _FIELD_SCOPE_DECLARED[cls] = declared
    return declared


_FREEZE_BUDGET = 256


class _Unfreezable(Exception):
    pass


def _freeze(value: Any, budget: list[int]) -> Any:
    value_type = type(value)
    if value_type is dict or value_type is list or value_type is tuple:
        budget[0] -= len(value)
        if budget[0] < 0:
            raise _Unfreezable
        if value_type is dict:
            return (
                dict,
                tuple((key, _freeze(item, budget)) for key, item in value.items()),
            )
        return (value_type, tuple(_freeze(item, budget) for item in value))
    if value_type is float:
        return (float, value.hex())
    if value_type in _IMMUTABLE_ARGUMENT_TYPES or isinstance(value, Enum):
        return (value_type, value)
    raise _Unfreezable


def _schema_context(
    schema: Any,
) -> tuple[Any, Any, dict[int, Any], frozenset[Any], OrderedDict[Any, Any]]:
    strawberry_schema = schema._strawberry_schema
    context = strawberry_schema.__dict__.get("_field_plan_context")
    if context is None:
        context = (
            strawberry_schema.schema_converter,
            strawberry_schema.config.info_class,
            {},
            frozenset(
                {schema.query_type, schema.mutation_type, schema.subscription_type}
            ),
            OrderedDict(),
        )
        strawberry_schema.__dict__["_field_plan_context"] = context
    return context


class FieldPlanExecutionContext(StrawberryGraphQLCoreExecutionContext):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._field_plans: dict[int, tuple[FieldDetailsList, _FieldPlan | None]] = {}
        (
            self._schema_converter,
            self._info_class,
            self._static_plans,
            self._root_types,
            self._plan_caches,
        ) = _schema_context(self.schema)
        self._shared_plans: dict[Any, tuple[Any, _FieldPlan | None]] | None = None
        cls = type(self)
        class_allowed = _CLASS_ALLOWED.get(cls)
        if class_allowed is None:
            class_allowed = all(_untouched(cls, name) for name in _BYPASSED_METHODS)
            _CLASS_ALLOWED[cls] = class_allowed
        self._shapes_allowed = (
            class_allowed and self.operation.operation != OperationType.SUBSCRIPTION
        )
        manager = self.middleware_manager
        self._middlewares: tuple[Any, ...] = (
            tuple(manager.middlewares) if manager is not None else ()
        )

    def execute_field(
        self,
        parent_type: GraphQLObjectType,
        source: Any,
        field_details_list: FieldDetailsList,
        path: Path,
        position_context: Any,
    ) -> Any:
        if position_context is not None or not self._shapes_allowed:
            return super().execute_field(
                parent_type, source, field_details_list, path, position_context
            )
        plan = self._get_plan(parent_type, field_details_list)
        if plan is None:
            return super().execute_field(
                parent_type, source, field_details_list, path, position_context
            )
        return self._execute_planned_field(
            plan, parent_type, source, field_details_list, path
        )

    def _get_plan(
        self, parent_type: GraphQLObjectType, field_details_list: FieldDetailsList
    ) -> _FieldPlan | None:
        cached = self._field_plans.get(id(field_details_list))
        if cached is not None:
            return cached[1]
        if _has_fragment_arguments(field_details_list):
            plan = self._build_plan(parent_type, field_details_list)
            self._field_plans[id(field_details_list)] = (field_details_list, plan)
            return plan
        shared_plans = self._shared_plans
        if shared_plans is None:
            shared_plans = self._shared_plans = self._request_plan_cache()
        nodes = tuple(details.node for details in field_details_list)
        node_key = (parent_type, tuple(map(id, nodes)))
        shared = shared_plans.get(node_key)
        if shared is None:
            plan = self._build_plan(parent_type, field_details_list)
            shared_plans[node_key] = (nodes, plan)
        else:
            plan = shared[1]
        self._field_plans[id(field_details_list)] = (field_details_list, plan)
        return plan

    def _request_plan_cache(self) -> dict[Any, tuple[Any, _FieldPlan | None]]:
        coerced = self.variable_values.coerced
        if coerced:
            try:
                variables = _freeze(coerced, [_FREEZE_BUDGET])
            except _Unfreezable:
                return {}
        else:
            variables = None
        key = (
            id(self.operation),
            type(self),
            tuple(type(middleware) for middleware in self._middlewares),
            variables,
        )
        caches = self._plan_caches
        with _PLAN_CACHE_LOCK:
            entry = caches.get(key)
            if entry is not None and entry[0]() is self.operation:
                caches.move_to_end(key)
                plans = entry[1]
                if plans is None:
                    plans = {}
                    caches[key] = (entry[0], plans)
                return plans
            caches[key] = (ref(self.operation), None)
            if len(caches) > _PLAN_CACHE_SIZE:
                caches.popitem(last=False)
        return {}

    def _coerce_leaf(self, leaf_shape: _Shape, value: Any) -> Any:
        if leaf_shape.enum_cache is not None:
            enum_class, names = leaf_shape.enum_cache
            if type(value) is enum_class:
                name = names.get(id(value))
                if name is None:
                    name = self.complete_leaf_value(leaf_shape.type_, value)
                    names[id(value)] = name
                return name
        return self.complete_leaf_value(leaf_shape.type_, value)

    def _execute_object(
        self,
        parent_type: GraphQLObjectType,
        source: Any,
        path: Path,
        entries: tuple[_ObjectEntry, ...],
        index: int | None = None,
    ) -> AwaitableOrValue[dict[str, Any]]:
        object_path = path if index is None else None
        item_index = 0 if index is None else index
        results: dict[str, Any] = {}
        is_awaitable = self.is_awaitable
        awaitable_fields: list[str] = []
        parent_type_name = parent_type.name
        try:
            for response_name, field_details_list, plan, getter in entries:
                if getter is not None:
                    python_name, exact_type, bound, nullable, list_mode = getter
                    try:
                        value = getattr(source, python_name)
                    except Exception as raw_error:  # noqa: BLE001
                        self.handle_field_error(
                            raw_error,
                            plan.return_type,
                            field_details_list,
                            Path(
                                (
                                    object_path
                                    if object_path is not None
                                    else (object_path := path.add_key(item_index, None))
                                ),
                                response_name,
                                parent_type_name,
                            ),
                        )
                        results[response_name] = None
                        continue
                    if list_mode == _COERCE_LEAF:
                        if (
                            value is not None
                            and value is not Undefined
                            and not isinstance(value, Exception)
                        ):
                            if is_awaitable(value):
                                results[response_name] = self._complete_awaitable_field(
                                    plan,
                                    parent_type,
                                    field_details_list,
                                    Path(
                                        (
                                            object_path
                                            if object_path is not None
                                            else (
                                                object_path := path.add_key(
                                                    item_index, None
                                                )
                                            )
                                        ),
                                        response_name,
                                        parent_type_name,
                                    ),
                                    value,
                                )
                                awaitable_fields.append(response_name)
                                continue
                            try:
                                completed = self._coerce_leaf(exact_type, value)
                            except Exception as raw_error:  # noqa: BLE001
                                self.handle_field_error(
                                    raw_error,
                                    plan.return_type,
                                    field_details_list,
                                    Path(
                                        (
                                            object_path
                                            if object_path is not None
                                            else (
                                                object_path := path.add_key(
                                                    item_index, None
                                                )
                                            )
                                        ),
                                        response_name,
                                        parent_type_name,
                                    ),
                                )
                                results[response_name] = None
                                continue
                            if is_awaitable(completed):
                                results[response_name] = self._await_completed(
                                    completed,
                                    plan.return_type,
                                    field_details_list,
                                    Path(
                                        (
                                            object_path
                                            if object_path is not None
                                            else (
                                                object_path := path.add_key(
                                                    item_index, None
                                                )
                                            )
                                        ),
                                        response_name,
                                        parent_type_name,
                                    ),
                                )
                                awaitable_fields.append(response_name)
                                continue
                            results[response_name] = completed
                            continue
                    elif list_mode == _NOT_A_LIST:
                        if type(value) is exact_type and (
                            bound == _UNBOUNDED
                            or (
                                GRAPHQL_MIN_INT <= value <= GRAPHQL_MAX_INT
                                if bound == _INT_BOUNDS
                                else isfinite(value)
                            )
                        ):
                            results[response_name] = value
                            continue
                    elif type(value) is list and _all_exact(
                        value, exact_type, bound, list_mode == _NULLABLE_ITEMS
                    ):
                        results[response_name] = value.copy()
                        continue
                    if value is None and nullable:
                        results[response_name] = None
                        continue
                    result = self._finish_planned_field(
                        plan,
                        parent_type,
                        field_details_list,
                        Path(
                            (
                                object_path
                                if object_path is not None
                                else (object_path := path.add_key(item_index, None))
                            ),
                            response_name,
                            parent_type_name,
                        ),
                        value,
                    )
                    results[response_name] = result
                    result_type = type(result)
                    if (
                        result_type is not dict
                        and result_type is not list
                        and is_awaitable(result)
                    ):
                        awaitable_fields.append(response_name)
                    continue
                field_path = Path(
                    (
                        object_path
                        if object_path is not None
                        else (object_path := path.add_key(item_index, None))
                    ),
                    response_name,
                    parent_type_name,
                )
                if plan is None:
                    result = super().execute_field(
                        parent_type, source, field_details_list, field_path, None
                    )
                else:
                    result = self._execute_planned_field(
                        plan, parent_type, source, field_details_list, field_path
                    )
                if result is not Undefined:
                    results[response_name] = result
                    result_type = type(result)
                    if (
                        result_type is not dict
                        and result_type is not list
                        and is_awaitable(result)
                    ):
                        awaitable_fields.append(response_name)
        except Exception:
            if awaitable_fields:
                self.settle_in_background(
                    [results[field] for field in awaitable_fields]
                )
            raise

        if not awaitable_fields:
            return results

        async def get_results() -> dict[str, Any]:
            if len(awaitable_fields) == 1:
                field = awaitable_fields[0]
                results[field] = await results[field]
            else:
                awaited_results = await gather_with_cancel(
                    *(results[field] for field in awaitable_fields)
                )
                results.update(zip(awaitable_fields, awaited_results, strict=True))
            return results

        return get_results()

    def _build_shape(
        self, type_: GraphQLOutputType, field_details_list: FieldDetailsList
    ) -> _Shape:
        if is_non_null_type(type_):
            child = self._build_shape(type_.of_type, field_details_list)
            kind = _NON_NULL_LEAF if child.kind == _LEAF else _NON_NULL
            return _Shape(kind, type_, child, None)
        if is_list_type(type_):
            return _Shape(
                _LIST,
                type_,
                self._build_shape(type_.of_type, field_details_list),
                None,
            )
        if is_leaf_type(type_):
            if (
                isinstance(type_, CustomGraphQLEnumType)
                and type(type_).coerce_output_value
                is CustomGraphQLEnumType.coerce_output_value
                and "coerce_output_value" not in vars(type_)
            ):
                return _Shape(_LEAF, type_, None, None, (type_.wrapped_cls, {}))
            return _Shape(_LEAF, type_, _scalar_fast_path(type_), None)
        if is_abstract_type(type_):
            return _Shape(_ABSTRACT, type_, {}, None)
        if is_object_type(type_):
            is_type_of = type_.is_type_of
            if is_type_of is not None and _code_of(is_type_of) not in _IS_TYPE_OF_CODES:
                return _Shape(_GENERIC, type_, None, None)
            grouped_field_set, new_defer_usages, _ = self.collect_subfields(
                type_, field_details_list
            )
            if new_defer_usages:
                return _Shape(_GENERIC, type_, None, None)
            entries = []
            for response_name, sub_field_details_list in grouped_field_set.items():
                sub_plan = self._get_plan(type_, sub_field_details_list)
                entries.append(
                    (
                        response_name,
                        sub_field_details_list,
                        sub_plan,
                        _inline_getter(sub_plan),
                    )
                )
            entries = tuple(entries)
            return _Shape(_OBJECT, type_, entries, is_type_of)
        return _Shape(_GENERIC, type_, None, None)

    def _build_plan(
        self,
        parent_type: GraphQLObjectType,
        field_details_list: FieldDetailsList,
    ) -> _FieldPlan | None:
        if any(
            directive.name.value == "stream"
            for details in field_details_list
            for directive in details.node.directives or ()
        ):
            return None
        field_name = field_details_list[0].node.name.value
        field_def = self.schema.get_field(parent_type, field_name)
        if field_def is None:
            return None
        if not self._middleware_applies(
            parent_type, field_def
        ) and not _has_fragment_arguments(field_details_list):
            static = self._static_plans.get(id(field_def))
            if static is not None:
                return static[1]
            plan = self._build_full_plan(
                parent_type, field_details_list, field_def, field_name
            )
            if plan is not None:
                return plan
        shape = self._build_shape(field_def.type, field_details_list)
        if parent_type in self._root_types and _is_static_shape(shape):
            return None
        return _FieldPlan(
            field_def,
            None,
            field_name,
            field_def.type,
            shape,
            _ORIGINAL_RESOLVER,
            False,
            (),
            None,
            [{}],
            getattr,
            "",
            _leaf_fast(shape),
        )

    def _middleware_applies(
        self, parent_type: GraphQLObjectType, field_def: GraphQLField
    ) -> bool:
        for middleware in self._middlewares:
            if not _declares_field_scope(
                type(middleware)
            ) or middleware.applies_to_field(parent_type, field_def):
                return True
        return False

    def _build_full_plan(
        self,
        parent_type: GraphQLObjectType,
        field_details_list: FieldDetailsList,
        field_def: GraphQLField,
        field_name: str,
    ) -> _FieldPlan | None:
        field = (field_def.extensions or {}).get(
            GraphQLCoreConverter.DEFINITION_BACKREF
        )
        if (
            not isinstance(field, StrawberryField)
            or type(field).get_result is not StrawberryField.get_result
            or field.extensions
            or field.is_subscription
        ):
            return None
        if self._schema_converter._get_field_exception_handlers(field):
            return None
        return_type = field_def.type

        if field.base_resolver is None:
            if (
                field_def.args
                or _code_of(field_def.resolve) not in _BASIC_RESOLVER_CODES
            ):
                return None
            return self._remember_static(
                _FieldPlan(
                    field_def,
                    field,
                    field_name,
                    return_type,
                    self._build_shape(return_type, field_details_list),
                    None,
                    False,
                    (),
                    None,
                    [{}],
                    field.default_resolver,
                    field.python_name,
                )
            )

        resolver = field.base_resolver
        if (
            (
                parent_type in self._root_types
                and (field_def.args or parent_type is not self.schema.query_type)
            )
            or self._info_class is not Info
            or _code_of(field_def.resolve) not in _RESOLVER_WRAPPER_CODES
            or type(resolver).__call__ is not StrawberryResolver.__call__
            or not callable(resolver.wrapped_func)
            or not all(
                _is_plannable_argument_type(argument.type)
                for argument in field_def.args.values()
            )
        ):
            return None
        kwargs: list[dict[str, Any] | None] = [None] if field_def.args else [{}]
        source_kwargs = tuple(
            parameter.name
            for parameter in (
                resolver.parent_parameter,
                resolver.root_parameter,
            )
            if parameter is not None
        )
        info_parameter = resolver.info_parameter
        return self._remember_static(
            _FieldPlan(
                field_def,
                field,
                field_name,
                return_type,
                self._build_shape(return_type, field_details_list),
                resolver.wrapped_func,
                resolver.self_parameter is not None,
                source_kwargs,
                info_parameter.name if info_parameter is not None else None,
                kwargs,
                field.default_resolver,
                field.python_name,
            )
        )

    def _remember_static(self, plan: _FieldPlan) -> _FieldPlan:
        plan = plan._replace(leaf_fast=_leaf_fast(plan.shape))
        if not plan.field_def.args and _is_static_shape(plan.shape):
            self._static_plans[id(plan.field_def)] = (plan.field_def, plan)
        return plan

    def _plan_info(
        self,
        plan: _FieldPlan,
        parent_type: GraphQLObjectType,
        path: Path,
        field_details_list: FieldDetailsList,
    ) -> GraphQLResolveInfo:
        return self.build_resolve_info(
            plan.field_def,
            [details.node for details in field_details_list],
            parent_type,
            path,
        )

    def _execute_planned_field(
        self,
        plan: _FieldPlan,
        parent_type: GraphQLObjectType,
        source: Any,
        field_details_list: FieldDetailsList,
        path: Path,
    ) -> AwaitableOrValue[Any]:
        if plan.resolver is _ORIGINAL_RESOLVER:
            return self._execute_with_original_resolver(
                plan, parent_type, source, field_details_list, path
            )
        resolver = plan.resolver
        try:
            if resolver is None:
                result = plan.default_resolver(source, plan.python_name)
            else:
                kwargs = plan.kwargs[0]
                if kwargs is None or kwargs is _CONVERT_PER_CALL:
                    first_details = field_details_list[0]
                    kwargs = convert_arguments(
                        get_argument_values(
                            plan.field_def,
                            first_details.node,
                            self.variable_values,
                            first_details.fragment_variable_values,
                            self.hide_suggestions,
                        ),
                        cast("StrawberryField", plan.field).arguments,
                        scalar_registry=self._schema_converter.scalar_registry,
                        config=self._schema_converter.config,
                    )
                    if plan.kwargs[0] is None:
                        plan.kwargs[0] = (
                            kwargs
                            if all(
                                type(value) in _IMMUTABLE_ARGUMENT_TYPES
                                or isinstance(value, Enum)
                                for value in kwargs.values()
                            )
                            else _CONVERT_PER_CALL
                        )
                if plan.source_kwargs or plan.info_kwarg:
                    kwargs = dict(kwargs)
                    for name in plan.source_kwargs:
                        kwargs[name] = source
                    if plan.info_kwarg:
                        kwargs[plan.info_kwarg] = self._info_class(
                            _raw_info=self._plan_info(
                                plan, parent_type, path, field_details_list
                            ),
                            _field=plan.field,
                        )
                result = (
                    resolver(source, **kwargs)
                    if plan.pass_source
                    else resolver(**kwargs)
                )
        except Exception as raw_error:  # noqa: BLE001
            self.handle_field_error(
                raw_error, plan.return_type, field_details_list, path
            )
            return None
        leaf_fast = plan.leaf_fast
        if leaf_fast is not None:
            exact_type, bound, nullable = leaf_fast
            if type(result) is exact_type and (
                bound == _UNBOUNDED
                or (
                    GRAPHQL_MIN_INT <= result <= GRAPHQL_MAX_INT
                    if bound == _INT_BOUNDS
                    else isfinite(result)
                )
            ):
                return result
            if result is None and nullable:
                return None
        return self._finish_planned_field(
            plan, parent_type, field_details_list, path, result
        )

    def _finish_planned_field(
        self,
        plan: _FieldPlan,
        parent_type: GraphQLObjectType,
        field_details_list: FieldDetailsList,
        path: Path,
        result: Any,
    ) -> AwaitableOrValue[Any]:
        try:
            if self.is_awaitable(result):
                return self._complete_awaitable_field(
                    plan, parent_type, field_details_list, path, result
                )
            completed = self._complete(
                plan.shape, plan, parent_type, field_details_list, path, path, result
            )
            if self.is_awaitable(completed):
                return self._await_completed(
                    completed, plan.return_type, field_details_list, path
                )
        except Exception as raw_error:  # noqa: BLE001
            self.handle_field_error(
                raw_error, plan.return_type, field_details_list, path
            )
            return None
        return completed

    def _execute_with_original_resolver(
        self,
        plan: _FieldPlan,
        parent_type: GraphQLObjectType,
        source: Any,
        field_details_list: FieldDetailsList,
        path: Path,
    ) -> AwaitableOrValue[Any]:
        field_def = plan.field_def
        resolve_fn = field_def.resolve or self.field_resolver
        if self.middleware_manager:
            resolve_fn = self.middleware_manager.get_field_resolver(resolve_fn)
        info = self._plan_info(plan, parent_type, path, field_details_list)
        try:
            first_details = field_details_list[0]
            args = get_argument_values(
                field_def,
                first_details.node,
                self.variable_values,
                first_details.fragment_variable_values,
                self.hide_suggestions,
            )
            result = resolve_fn(source, info, **args)
            leaf_fast = plan.leaf_fast
            if leaf_fast is not None:
                exact_type, bound, nullable = leaf_fast
                if type(result) is exact_type and (
                    bound == _UNBOUNDED
                    or (
                        GRAPHQL_MIN_INT <= result <= GRAPHQL_MAX_INT
                        if bound == _INT_BOUNDS
                        else isfinite(result)
                    )
                ):
                    return result
                if result is None and nullable:
                    return None
            if self.is_awaitable(result):
                return self._complete_awaitable_field(
                    plan, parent_type, field_details_list, path, result
                )
            completed = self._complete(
                plan.shape, plan, parent_type, field_details_list, path, path, result
            )
            if self.is_awaitable(completed):
                return self._await_completed(
                    completed, plan.return_type, field_details_list, path
                )
        except Exception as raw_error:  # noqa: BLE001
            self.handle_field_error(
                raw_error, plan.return_type, field_details_list, path
            )
            return None
        return completed

    async def _complete_awaitable_field(
        self,
        plan: _FieldPlan,
        parent_type: GraphQLObjectType,
        field_details_list: FieldDetailsList,
        path: Path,
        result: Any,
    ) -> Any:
        try:
            resolved = await (
                result if self.abort_signal is None else self.with_abort_signal(result)
            )
            completed = self._complete(
                plan.shape,
                plan,
                parent_type,
                field_details_list,
                path,
                path,
                resolved,
            )
            if self.is_awaitable(completed):
                completed = await completed
        except Exception as raw_error:  # noqa: BLE001
            self.handle_field_error(
                raw_error, plan.return_type, field_details_list, path
            )
            completed = None
        return completed

    async def _await_completed(
        self,
        completed: Any,
        return_type: GraphQLOutputType,
        field_details_list: FieldDetailsList,
        path: Path,
    ) -> Any:
        try:
            return await completed
        except Exception as raw_error:  # noqa: BLE001
            self.handle_field_error(raw_error, return_type, field_details_list, path)
            return None

    def _complete(  # noqa: PLR0917
        self,
        shape: _Shape,
        plan: _FieldPlan,
        parent_type: GraphQLObjectType,
        field_details_list: FieldDetailsList,
        field_path: Path,
        path: Path,
        result: Any,
        info: GraphQLResolveInfo | None = None,
        index: int | None = None,
    ) -> AwaitableOrValue[Any]:
        if isinstance(result, Exception):
            raise result
        kind = shape.kind
        if kind in (_NON_NULL_LEAF, _NON_NULL):
            if result is None or result is Undefined:
                msg = (
                    "Cannot return null for non-nullable field"
                    f" {parent_type}.{plan.field_name}."
                )
                raise TypeError(msg)
            shape = shape.child
            kind = shape.kind
        elif result is None or result is Undefined:
            return None
        if kind == _LEAF:
            exact_type = shape.child
            result_type = type(result)
            if exact_type is not None:
                if (
                    result_type is exact_type
                    and (
                        exact_type is not int
                        or GRAPHQL_MIN_INT <= result <= GRAPHQL_MAX_INT
                    )
                    and (exact_type is not float or isfinite(result))
                ):
                    return result
            elif shape.enum_cache is not None:
                enum_class, names = shape.enum_cache
                if result_type is enum_class:
                    name = names.get(id(result))
                    if name is None:
                        name = self.complete_leaf_value(shape.type_, result)
                        names[id(result)] = name
                    return name
            return self.complete_leaf_value(shape.type_, result)
        if kind == _OBJECT:
            if shape.is_type_of is not None and not shape.is_type_of(result, None):
                raise invalid_return_type_error(shape.type_, result, field_details_list)
            return self._execute_object(shape.type_, result, path, shape.child, index)
        if index is not None:
            path = path.add_key(index, None)
        if kind == _LIST:
            return self._complete_list(
                shape,
                plan,
                parent_type,
                field_details_list,
                field_path,
                path,
                result,
            )
        if info is None:
            info = self._plan_info(plan, parent_type, field_path, field_details_list)
        if kind == _ABSTRACT:
            return self._complete_abstract(
                shape, field_details_list, info, path, result
            )
        return self.complete_value(
            shape.type_, field_details_list, info, path, result, None
        )

    def _complete_abstract(
        self,
        shape: _Shape,
        field_details_list: FieldDetailsList,
        info: GraphQLResolveInfo,
        path: Path,
        result: Any,
    ) -> AwaitableOrValue[Any]:
        return_type = shape.type_
        resolve_type_fn = return_type.resolve_type or self.type_resolver
        runtime_type = resolve_type_fn(result, info, return_type)
        if self.is_awaitable(runtime_type):

            async def await_complete_object_value() -> Any:
                value = self.complete_object_value(
                    self.ensure_valid_runtime_type(
                        await self.with_abort_signal(runtime_type),
                        return_type,
                        field_details_list,
                        info,
                        result,
                    ),
                    field_details_list,
                    info,
                    path,
                    result,
                    None,
                )
                if self.is_awaitable(value):
                    return await value
                return value

            return await_complete_object_value()
        object_type = self.ensure_valid_runtime_type(
            runtime_type, return_type, field_details_list, info, result
        )
        object_shapes = shape.child
        object_shape = object_shapes.get(object_type)
        if object_shape is None:
            object_shapes[object_type] = _SEEN_ONCE
            return self.complete_object_value(
                object_type, field_details_list, info, path, result, None
            )
        if object_shape is _SEEN_ONCE:
            object_shape = self._build_shape(object_type, field_details_list)
            object_shapes[object_type] = object_shape
        if object_shape.kind != _OBJECT:
            return self.complete_object_value(
                object_type, field_details_list, info, path, result, None
            )
        if object_shape.is_type_of is not None and not object_shape.is_type_of(
            result, None
        ):
            raise invalid_return_type_error(object_type, result, field_details_list)
        return self._execute_object(object_type, result, path, object_shape.child)

    def _complete_list(  # noqa: PLR0917
        self,
        shape: _Shape,
        plan: _FieldPlan,
        parent_type: GraphQLObjectType,
        field_details_list: FieldDetailsList,
        field_path: Path,
        path: Path,
        result: Any,
    ) -> AwaitableOrValue[list[Any]]:
        item_shape = shape.child
        native_sequence = type(result) in _NATIVE_SEQUENCES
        if (
            item_shape.kind == _GENERIC
            or (item_shape.kind == _NON_NULL and item_shape.child.kind == _GENERIC)
            or (not native_sequence and self.is_async_iterable(result))
        ):
            return self.complete_list_value(
                shape.type_,
                field_details_list,
                self._plan_info(plan, parent_type, field_path, field_details_list),
                path,
                result,
                None,
            )
        if not native_sequence and not is_iterable(result):
            msg = (
                "Expected Iterable, but did not find one for field"
                f" '{parent_type}.{plan.field_name}'."
            )
            raise GraphQLError(msg)
        item_type = shape.type_.of_type
        is_awaitable = self.is_awaitable
        completed_results: list[Any] = []
        append_completed = completed_results.append
        awaitable_indices: list[int] = []
        info: GraphQLResolveInfo | None = None
        leaf_shape = (
            item_shape
            if item_shape.kind == _LEAF
            else item_shape.child
            if item_shape.kind == _NON_NULL_LEAF
            else None
        )
        exact_type = None if leaf_shape is None else leaf_shape.child
        bounded = exact_type is int or exact_type is float
        abstract_items = item_shape.kind == _ABSTRACT or (
            item_shape.kind == _NON_NULL and item_shape.child.kind == _ABSTRACT
        )
        iterator = iter(result)
        index = 0
        try:
            for item in iterator:
                if type(item) is exact_type and (
                    not bounded
                    or (
                        GRAPHQL_MIN_INT <= item <= GRAPHQL_MAX_INT
                        if exact_type is int
                        else isfinite(item)
                    )
                ):
                    append_completed(item)
                    index += 1
                    continue
                if is_awaitable(item):
                    item_path = path.add_key(index, None)
                    if info is None:
                        info = self._plan_info(
                            plan, parent_type, field_path, field_details_list
                        )
                    append_completed(
                        self.complete_awaitable_list_item_value(
                            item,
                            item_type,
                            field_details_list,
                            info,
                            item_path,
                            None,
                        )
                    )
                    awaitable_indices.append(index)
                else:
                    try:
                        if info is None and abstract_items:
                            info = self._plan_info(
                                plan, parent_type, field_path, field_details_list
                            )
                        completed_item = self._complete(
                            item_shape,
                            plan,
                            parent_type,
                            field_details_list,
                            field_path,
                            path,
                            item,
                            info,
                            index,
                        )
                        if is_awaitable(completed_item):
                            append_completed(
                                self._await_completed(
                                    completed_item,
                                    item_type,
                                    field_details_list,
                                    path.add_key(index, None),
                                )
                            )
                            awaitable_indices.append(index)
                        else:
                            append_completed(completed_item)
                    except Exception as raw_error:  # noqa: BLE001
                        self.handle_field_error(
                            raw_error,
                            item_type,
                            field_details_list,
                            path.add_key(index, None),
                        )
                        append_completed(None)
                index += 1
        except Exception:
            maybe_awaitables = [completed_results[index] for index in awaitable_indices]
            maybe_awaitables.extend(collect_iterator_awaitables(iterator, is_awaitable))
            if maybe_awaitables:
                self.settle_in_background(maybe_awaitables)
            raise

        if not awaitable_indices:
            return completed_results

        async def get_completed_results() -> list[Any]:
            if len(awaitable_indices) == 1:
                index = awaitable_indices[0]
                completed_results[index] = await completed_results[index]
            else:
                awaited_results = await gather_with_cancel(
                    *(completed_results[index] for index in awaitable_indices)
                )
                for index, sub_result in zip(
                    awaitable_indices, awaited_results, strict=True
                ):
                    completed_results[index] = sub_result
            return completed_results

        return get_completed_results()
