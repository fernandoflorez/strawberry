"""Tests for FieldPlanExecutionContext.

Every behavioural test runs the same operation with Strawberry's default
execution context and with FieldPlanExecutionContext and asserts that the
observable result is identical: data, errors (message, path, locations and
order) and the side effects recorded by resolvers, middleware and scalars.
Other tests check that the fast paths are really taken and that the executor
falls back to the default path whenever a shortcut could change behaviour.
"""

import asyncio
import enum
import functools
import json
import math
from collections.abc import AsyncGenerator
from typing import Annotated, Any, Generic, NewType, TypeVar

import pytest
from graphql import GraphQLField, GraphQLObjectType

import strawberry
from strawberry.dataloader import DataLoader
from strawberry.execution.field_plan import FieldPlanExecutionContext
from strawberry.extensions import (
    FieldExtension,
    ParserCache,
    SchemaExtension,
    ValidationCache,
)
from strawberry.extensions.tracing import ApolloTracingExtensionSync
from strawberry.permission import BasePermission
from strawberry.schema.config import StrawberryConfig
from strawberry.types import Info
from strawberry.types.fields.resolver import StrawberryResolver

StrawberryGraphQLCoreExecutionContext = FieldPlanExecutionContext.__bases__[0]

T = TypeVar("T")

EVENTS: list[str] = []


@strawberry.enum
class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = 3


def _serialize_upper(value: Any) -> str:
    EVENTS.append(f"serialize {value}")
    return str(value).upper()


Upper = strawberry.scalar(
    NewType("Upper", str),
    serialize=_serialize_upper,
    parse_value=lambda v: v,
)

NullScalar = strawberry.scalar(
    NewType("NullScalar", str),
    serialize=lambda v: None,
    parse_value=lambda v: v,
)


class Boom(Exception):
    pass


@strawberry.interface
class Node:
    id: strawberry.ID


@strawberry.type
class Leaf(Node):
    id: strawberry.ID
    name: str
    maybe: str | None = None
    numbers: list[int | None] = strawberry.field(default_factory=list)


@strawberry.type
class Branch(Node):
    id: strawberry.ID
    size: int


@strawberry.type
class Other:
    value: float


Thing = Annotated[Leaf | Other, strawberry.union("Thing")]


class Deny(BasePermission):
    message = "denied"

    def has_permission(self, source: Any, info: Info, **kwargs: Any) -> bool:
        return False


class Suffix(FieldExtension):
    def resolve(self, next_: Any, source: Any, info: Info, **kwargs: Any) -> Any:
        return f"{next_(source, info, **kwargs)}!"


@strawberry.input
class Filter:
    prefix: str = "x"
    limit: int = 2


@strawberry.type
class Box(Generic[T]):
    item: T
    items: list[T]


class Weird:
    @property
    def name(self) -> str:
        raise Boom("property boom")


@strawberry.type
class Fragile:
    index: int
    label: str
    count: int


class FragileSource:
    def __init__(self, index: int) -> None:
        self.index = index

    @property
    def label(self) -> str:
        if self.index == 1:
            raise Boom("fragile label")
        return f"f{self.index}"

    @property
    def count(self) -> int:
        return 2**33 if self.index == 2 else self.index


async def load_squares(keys: list[int]) -> list[int | Exception]:
    EVENTS.append(f"batch {keys}")
    return [k * k if k != 13 else Boom("unlucky") for k in keys]


@strawberry.type
class Item:
    index: int
    label: str
    color: Color
    ratio: float
    flag: bool
    upper: Upper
    tags: list[str]
    maybe_tags: list[str | None] | None
    nested: list[list[int]]
    scores: list[int]
    weights: list[float]
    nulled: NullScalar | None
    big: int
    loose_color: Color

    @strawberry.field
    def doubled(self) -> int:
        return self.index * 2

    @strawberry.field
    def greet(self, name: str = "you", times: int = 1) -> str:
        return f"{self.label}:{name * times}"

    @strawberry.field
    def by_color(self, color: Color = Color.RED) -> str:
        return f"{self.index}-{color.name}"

    @strawberry.field
    def with_info(self, info: Info) -> str:
        return f"{info.field_name}@{info.path.key}"

    @strawberry.field
    async def later(self) -> int:
        await asyncio.sleep(0)
        return self.index + 100

    @strawberry.field
    async def square(self, info: Info) -> int:
        return await info.context["loader"].load(self.index)

    @strawberry.field
    def failing(self) -> str | None:
        if self.index % 2:
            raise Boom(f"fail {self.index}")
        return "ok"

    @strawberry.field
    def strict_failing(self) -> str:
        if self.index == 3:
            raise Boom("strict fail")
        return "fine"

    @strawberry.field
    def null_in_non_null(self) -> str:
        return None if self.index == 2 else "x"  # type: ignore[return-value]

    @strawberry.field
    def bad_int(self) -> int:
        return 2**40 if self.index == 1 else self.index

    @strawberry.field
    def bad_float(self) -> float | None:
        return math.nan if self.index == 1 else 1.5

    @strawberry.field
    def returns_exception(self) -> str | None:
        return Boom("returned") if self.index == 0 else "fine"  # type: ignore[return-value]

    @strawberry.field(permission_classes=[Deny])
    def secret(self) -> str | None:
        return "secret"

    @strawberry.field(extensions=[Suffix()])
    def suffixed(self) -> str:
        return self.label

    @strawberry.field
    def filtered(self, filter: Filter | None = None) -> list[str]:
        f = filter or Filter()
        return [f"{f.prefix}{i}" for i in range(f.limit)]

    @strawberry.field
    def picked(self, ids: list[int]) -> list[int]:
        return [i * self.index for i in ids]

    @strawberry.field
    def node(self) -> Node:
        if self.index % 2:
            return Leaf(id=strawberry.ID(f"L{self.index}"), name="n")
        return Branch(id=strawberry.ID(f"B{self.index}"), size=self.index)

    @strawberry.field
    def thing(self) -> Thing:
        if self.index % 2:
            return Leaf(id=strawberry.ID("t"), name="leaf")
        return Other(value=self.index / 2)

    @strawberry.field
    def box(self) -> Box[int]:
        return Box(item=self.index, items=[self.index, self.index + 1])

    @strawberry.field
    def wrong_type(self) -> Leaf | None:
        return Other(value=1.0) if self.index == 4 else None  # type: ignore[return-value]


def make_item(i: int) -> Item:
    return Item(
        index=i,
        label=f"item{i}",
        color=list(Color)[i % 3],
        ratio=i / 3,
        flag=bool(i % 2),
        upper=Upper(f"up{i}"),
        tags=[f"t{i}", f"u{i}"],
        maybe_tags=None if i == 2 else ["a", None, "b"],
        nested=[[i, i + 1], []],
        scores=[i, 2**40] if i == 3 else [i, i + 1],
        weights=[math.inf] if i == 1 else [0.5, float(i)],
        nulled=NullScalar("n") if i == 4 else None,  # type: ignore[arg-type]
        big=2**40 if i == 3 else i,
        loose_color="red" if i == 1 else Color.BLUE,  # type: ignore[arg-type]
    )


@strawberry.type
class Query:
    @strawberry.field
    def items(self, count: int = 5) -> list[Item]:
        return [make_item(i) for i in range(count)]

    @strawberry.field
    def maybe_items(self) -> list[Item | None] | None:
        return [make_item(0), None, make_item(1)]

    @strawberry.field
    def strict_items(self) -> list[Item]:
        return [make_item(0), None, make_item(1)]  # type: ignore[list-item]

    @strawberry.field
    async def async_items(self, count: int = 3) -> list[Item]:
        return [make_item(i) for i in range(count)]

    @strawberry.field
    def item(self, index: int) -> Item | None:
        return make_item(index) if index >= 0 else None

    @strawberry.field
    def weird(self) -> Leaf | None:
        return Weird()  # type: ignore[return-value]

    @strawberry.field
    def nodes(self) -> list[Node]:
        return [
            Leaf(id=strawberry.ID("1"), name="a", numbers=[1, None]),
            Branch(id=strawberry.ID("2"), size=3),
            Leaf(id=strawberry.ID("3"), name="b"),
            Branch(id=strawberry.ID("4"), size=4),
        ]

    @strawberry.field
    def things(self) -> list[Thing]:
        return [Leaf(id=strawberry.ID("x"), name="x"), Other(value=2.5)]

    @strawberry.field
    def color(self, color: Color) -> Color:
        return color

    @strawberry.field
    def not_a_list(self) -> list[int] | None:
        return 5  # type: ignore[return-value]

    @strawberry.field
    def generator_list(self) -> list[int]:
        return (i for i in range(3))  # type: ignore[return-value]

    @strawberry.field
    def fragile(self) -> list[Fragile | None]:
        return [FragileSource(i) for i in range(4)]  # type: ignore[misc]


@strawberry.type
class Mutation:
    @strawberry.mutation
    def bump(self, value: int) -> Item:
        EVENTS.append(f"bump {value}")
        return make_item(value)


class Observe(SchemaExtension):
    def resolve(
        self, _next: Any, root: Any, info: Any, *args: Any, **kwargs: Any
    ) -> Any:
        EVENTS.append(f"observe {info.parent_type.name}.{info.field_name}")
        return _next(root, info, *args, **kwargs)


class ObserveRootOnly(Observe):
    def applies_to_field(
        self, parent_type: GraphQLObjectType, field_def: GraphQLField
    ) -> bool:
        return parent_type.name == "Query"

    def resolve(
        self, _next: Any, root: Any, info: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if info.parent_type.name != "Query":
            return _next(root, info, *args, **kwargs)
        return super().resolve(_next, root, info, *args, **kwargs)


ITEM_FIELDS = """
  index label color ratio flag upper tags maybeTags nested scores weights nulled
  big looseColor
  doubled greet greetTwice: greet(name: "bo", times: 2) greetVar: greet(name: $name)
  byColor byBlue: byColor(color: BLUE) withInfo later square failing strictFailing
  badFloat returnsException secret suffixed
  filtered filteredCustom: filtered(filter: {prefix: "p", limit: 3})
  picked(ids: [1, 2])
  node { __typename id ... on Leaf { name maybe numbers } ... on Branch { size } }
  thing { __typename ... on Leaf { id name } ... on Other { value } }
  box { item items }
  wrongType { id }
"""

QUERIES: dict[str, tuple[str, dict[str, Any]]] = {
    "items": (
        f'query Q($name: String = "z") {{ items {{ {ITEM_FIELDS} }} }}',
        {},
    ),
    "items_vars": (
        f"query Q($name: String) {{ items(count: 4) {{ {ITEM_FIELDS} }} }}",
        {"name": "vv"},
    ),
    "null_non_null": ("{ items(count: 4) { index nullInNonNull } }", {}),
    "bad_int": ("{ items(count: 3) { index badInt } }", {}),
    "maybe_items": ("{ maybeItems { index label failing } }", {}),
    "strict_items": ("{ strictItems { index label } }", {}),
    "async_items": (
        f"query Q($name: String) {{ asyncItems {{ {ITEM_FIELDS} }} }}",
        {},
    ),
    "item_null": (
        "{ item(index: -1) { index } a: item(index: 2) { index label } }",
        {},
    ),
    "weird": ("{ weird { name } }", {}),
    "nodes": (
        (
            "{ nodes { __typename id ... on Leaf { name numbers } "
            "... on Branch { size } } }"
        ),
        {},
    ),
    "things": (
        "{ things { __typename ... on Leaf { id name } ... on Other { value } } }",
        {},
    ),
    "fragments": (
        (
            "fragment F on Item { index label } "
            "{ items(count: 2) { ...F ... on Item { color } alias: label } }"
        ),
        {},
    ),
    "include_skip": (
        (
            "query Q($yes: Boolean!, $no: Boolean!) { items(count: 2) { index "
            "label @include(if: $yes) color @skip(if: $yes) flag @include(if: $no) } }"
        ),
        {"yes": True, "no": False},
    ),
    "enum_arg": (
        "query Q($c: Color!) { color(color: $c) other: color(color: GREEN) }",
        {"c": "BLUE"},
    ),
    "typename": ("{ __typename items(count: 1) { __typename } }", {}),
    "not_a_list": ("{ notAList }", {}),
    "generator_list": ("{ generatorList }", {}),
    "same_field_args": (
        '{ items(count: 2) { a: greet(name: "a") b: greet(name: "b") c: greet } }',
        {},
    ),
    "mutation": (
        "mutation { a: bump(value: 1) { index label } b: bump(value: 2) { doubled } }",
        {},
    ),
    "strict_failing": ("{ items(count: 5) { index strictFailing } }", {}),
    "fragile": ("{ fragile { index label count } }", {}),
    "dataloader_errors": ("{ items(count: 15) { index square } }", {}),
}

ASYNC_ONLY = {"items", "items_vars", "async_items", "dataloader_errors"}

EXTENSIONS: dict[str, list[Any]] = {
    "no_extensions": [],
    "middleware": [Observe],
    "root_only_middleware": [ObserveRootOnly],
    "apollo_tracing_sync": [ApolloTracingExtensionSync],
}


def build_schema(
    execution_context_class: type | None,
    extensions: list[Any],
    config: StrawberryConfig | None = None,
) -> strawberry.Schema:
    return strawberry.Schema(
        query=Query,
        mutation=Mutation,
        types=[Leaf, Branch],
        execution_context_class=execution_context_class,
        extensions=extensions,
        config=config,
    )


def snapshot(result: Any) -> str:
    return json.dumps(
        {
            "data": result.data,
            "errors": [error.formatted for error in result.errors or []],
            "events": list(EVENTS),
        },
        default=str,
    )


async def run_query(
    schema: strawberry.Schema, query: str, variables: dict[str, Any], *, sync: bool
) -> str:
    EVENTS.clear()
    context = {"loader": DataLoader(load_fn=load_squares)}
    if sync:
        result = schema.execute_sync(
            query, variable_values=variables, context_value=context
        )
    else:
        result = await schema.execute(
            query, variable_values=variables, context_value=context
        )
    return snapshot(result)


async def assert_same(
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    extensions: list[Any] | None = None,
    sync: bool = False,
    base_class: type = StrawberryGraphQLCoreExecutionContext,
    plan_class: type = FieldPlanExecutionContext,
    config: StrawberryConfig | None = None,
) -> str:
    extensions = extensions or []
    variables = variables or {}
    base = await run_query(
        build_schema(base_class, extensions, config), query, variables, sync=sync
    )
    plan = await run_query(
        build_schema(plan_class, extensions, config), query, variables, sync=sync
    )
    assert plan == base
    return plan


@pytest.mark.parametrize("extensions", EXTENSIONS.values(), ids=EXTENSIONS.keys())
@pytest.mark.parametrize("name", QUERIES.keys())
async def test_matches_default_executor_async(name: str, extensions: list[Any]) -> None:
    query, variables = QUERIES[name]
    await assert_same(query, variables, extensions=extensions)


@pytest.mark.parametrize("extensions", EXTENSIONS.values(), ids=EXTENSIONS.keys())
@pytest.mark.parametrize("name", sorted(set(QUERIES) - ASYNC_ONLY))
async def test_matches_default_executor_sync(name: str, extensions: list[Any]) -> None:
    query, variables = QUERIES[name]
    await assert_same(query, variables, extensions=extensions, sync=True)


async def test_plans_are_used_for_nested_fields() -> None:
    query = "{ items(count: 3) { index greet node { id } } }"
    schema = build_schema(FieldPlanExecutionContext, [])
    calls: list[str] = []
    original = FieldPlanExecutionContext._execute_planned_field

    def spy(self: Any, plan: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(plan.field_name)
        return original(self, plan, *args, **kwargs)

    FieldPlanExecutionContext._execute_planned_field = spy  # type: ignore[method-assign]
    try:
        result = await schema.execute(query)
    finally:
        FieldPlanExecutionContext._execute_planned_field = original  # type: ignore[method-assign]
    assert result.errors is None
    assert "greet" in calls
    assert "items" in calls


async def test_info_is_only_built_when_requested() -> None:
    built: list[str] = []

    class Tracking(FieldPlanExecutionContext):
        pass

    original = StrawberryGraphQLCoreExecutionContext.build_resolve_info

    def build_resolve_info(self: Any, field_def: Any, field_nodes: Any, *a: Any) -> Any:
        built.append(field_nodes[0].name.value)
        return original(self, field_def, field_nodes, *a)

    schema = build_schema(FieldPlanExecutionContext, [])
    StrawberryGraphQLCoreExecutionContext.build_resolve_info = build_resolve_info  # type: ignore[method-assign]
    try:
        result = await schema.execute("{ items(count: 2) { greet withInfo } }")
    finally:
        StrawberryGraphQLCoreExecutionContext.build_resolve_info = original  # type: ignore[method-assign]
    assert result.errors is None
    assert "greet" not in built
    assert built.count("withInfo") == 2


async def test_leaf_lists_are_copied() -> None:
    shared = ["a", "b"]

    @strawberry.type
    class Holder:
        tags: list[str]

    @strawberry.type
    class ListQuery:
        @strawberry.field
        def holders(self) -> list[Holder]:
            return [Holder(tags=shared), Holder(tags=shared)]

    schema = strawberry.Schema(
        query=ListQuery, execution_context_class=FieldPlanExecutionContext
    )
    result = await schema.execute("{ holders { tags } }")
    assert result.data == {"holders": [{"tags": ["a", "b"]}, {"tags": ["a", "b"]}]}
    first = result.data["holders"][0]["tags"]
    assert first is not shared
    first.append("c")
    assert shared == ["a", "b"]


async def test_list_subclasses_take_the_default_path() -> None:
    class Tags(list):
        pass

    @strawberry.type
    class Holder:
        tags: list[str]

    @strawberry.type
    class ListQuery:
        @strawberry.field
        def holder(self) -> Holder:
            return Holder(tags=Tags(["x"]))

    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = strawberry.Schema(query=ListQuery, execution_context_class=cls)
        result = await schema.execute("{ holder { tags } }")
        assert result.data == {"holder": {"tags": ["x"]}}


@pytest.mark.parametrize(
    "variables",
    [
        {"name": "a", "n": 2, "c": "RED", "yes": True},
        {"name": "b", "n": 3, "c": "BLUE", "yes": False},
        {"name": None, "n": 1, "c": "GREEN", "yes": True},
    ],
)
async def test_plan_cache_with_alternating_variables(
    variables: dict[str, Any],
) -> None:
    query = (
        "query Q($name: String, $n: Int!, $c: Color!, $yes: Boolean!) "
        "{ items(count: $n) { index greet(name: $name) byColor(color: $c) "
        "label @include(if: $yes) color @skip(if: $yes) } }"
    )
    others = [
        {"name": "z", "n": 4, "c": "BLUE", "yes": False},
        {"name": "y", "n": 1, "c": "RED", "yes": True},
    ]
    extensions = [ParserCache(maxsize=10), ValidationCache(maxsize=10)]
    base_schema = build_schema(StrawberryGraphQLCoreExecutionContext, extensions)
    plan_schema = build_schema(FieldPlanExecutionContext, extensions)
    for current in [variables, *others, variables, *others, variables]:
        base = await run_query(base_schema, query, current, sync=False)
        plan = await run_query(plan_schema, query, current, sync=False)
        assert plan == base


async def test_plan_cache_distinguishes_equal_values_of_different_types() -> None:
    @strawberry.type
    class EchoQuery:
        @strawberry.field
        def echo(self, value: float) -> str:
            return repr(value)

    extensions = [ParserCache(maxsize=10)]
    query = "query Q($v: Float!) { echo(value: $v) }"
    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = strawberry.Schema(
            query=EchoQuery, extensions=extensions, execution_context_class=cls
        )
        results = [
            (await schema.execute(query, variable_values={"v": value})).data
            for value in (1, 1.0, 1, 1.0)
        ]
        assert results == [{"echo": "1.0"}] * 4 or results[0] == results[2]


async def test_plan_cache_is_reused_for_the_same_document() -> None:
    extensions = [ParserCache(maxsize=10)]
    schema = build_schema(FieldPlanExecutionContext, extensions)
    builds: list[str] = []
    original = FieldPlanExecutionContext._build_plan

    def spy(self: Any, parent_type: Any, field_details_list: Any) -> Any:
        builds.append(field_details_list[0].node.name.value)
        return original(self, parent_type, field_details_list)

    FieldPlanExecutionContext._build_plan = spy  # type: ignore[method-assign]
    try:
        counts = []
        for _ in range(4):
            builds.clear()
            await schema.execute("{ items(count: 2) { index greet } }")
            counts.append(len(builds))
    finally:
        FieldPlanExecutionContext._build_plan = original  # type: ignore[method-assign]
    assert counts[0] > 0
    assert counts[2] == 0
    assert counts[3] == 0


async def test_large_variables_are_not_cached_but_still_work() -> None:
    query = "query Q($ids: [Int!]!) { items(count: 1) { picked(ids: $ids) } }"
    variables = {"ids": list(range(1000))}
    await assert_same(query, variables, extensions=[ParserCache(maxsize=10)])


async def test_subclass_overriding_completion_disables_plans() -> None:
    class CustomAbstract(StrawberryGraphQLCoreExecutionContext):
        def complete_abstract_value(self, *args: Any, **kwargs: Any) -> Any:
            EVENTS.append("complete_abstract_value")
            return super().complete_abstract_value(*args, **kwargs)

    class PlanCustomAbstract(FieldPlanExecutionContext):
        def complete_abstract_value(self, *args: Any, **kwargs: Any) -> Any:
            EVENTS.append("complete_abstract_value")
            return super().complete_abstract_value(*args, **kwargs)

    result = await assert_same(
        "{ nodes { id } things { __typename } }",
        base_class=CustomAbstract,
        plan_class=PlanCustomAbstract,
    )
    assert "complete_abstract_value" in result


async def test_wrapped_resolvers_are_called() -> None:
    def traced(resolver: Any) -> Any:
        @functools.wraps(resolver)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            EVENTS.append("traced")
            return resolver(*args, **kwargs)

        return wrapper

    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = build_schema(cls, [])
        item_type = schema._schema.type_map["Item"]
        for field_name in ("label", "greet"):
            field_def = item_type.fields[field_name]
            field_def.resolve = traced(field_def.resolve)
        EVENTS.clear()
        result = await schema.execute("{ items(count: 2) { label greet } }")
        assert result.errors is None
        assert EVENTS.count("traced") == 4


async def test_resolver_subclass_call_is_respected() -> None:
    class LoudResolver(StrawberryResolver):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            EVENTS.append("loud")
            return super().__call__(*args, **kwargs)

    def shout() -> str:
        return "x"

    @strawberry.type
    class LoudParent:
        value: str = strawberry.field(resolver=LoudResolver(shout))

    @strawberry.type
    class LoudQuery:
        @strawberry.field
        def items(self) -> list[LoudParent]:
            return [LoudParent(), LoudParent()]

    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = strawberry.Schema(query=LoudQuery, execution_context_class=cls)
        EVENTS.clear()
        result = await schema.execute("{ items { value } }")
        assert result.data == {"items": [{"value": "x"}, {"value": "x"}]}
        assert EVENTS.count("loud") == 2


async def test_custom_info_class_is_built_like_the_default_executor() -> None:
    class CountingInfo(Info):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            EVENTS.append("info")
            super().__init__(*args, **kwargs)

    await assert_same(
        "{ items(count: 3) { greet doubled } }",
        config=StrawberryConfig(info_class=CountingInfo),
    )


async def test_patched_base_methods_disable_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strawberry.execution import field_plan

    original = StrawberryGraphQLCoreExecutionContext.complete_value

    @functools.wraps(original)
    def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        EVENTS.append("complete_value")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(StrawberryGraphQLCoreExecutionContext, "complete_value", traced)
    monkeypatch.setattr(field_plan, "_CLASS_ALLOWED", {})
    result = await assert_same("{ items(count: 2) { index label } }")
    assert "complete_value" in result


async def test_middleware_without_hook_wraps_every_field() -> None:
    result = await assert_same(
        "{ items(count: 2) { index label greet } }", extensions=[Observe]
    )
    assert "observe Item.index" in result


async def test_middleware_hook_skips_nested_fields() -> None:
    result = await assert_same(
        "{ items(count: 2) { index label greet } }", extensions=[ObserveRootOnly]
    )
    assert "observe Query.items" in result
    assert "observe Item.index" not in result


def test_tracing_sync_extension_applies_only_to_custom_resolvers() -> None:
    schema = build_schema(None, [])
    item_type = schema._schema.type_map["Item"]
    extension = ApolloTracingExtensionSync()
    assert not extension.applies_to_field(item_type, item_type.fields["label"])
    assert extension.applies_to_field(item_type, item_type.fields["greet"])


async def test_abstract_types_with_many_runtime_types() -> None:
    @strawberry.interface
    class Animal:
        name: str

    types = [strawberry.type(type(f"Animal{i}", (Animal,), {})) for i in range(6)]

    @strawberry.type
    class ZooQuery:
        @strawberry.field
        def animals(self) -> list[Animal]:
            return [types[i % 6](name=f"a{i}") for i in range(20)]

    for sync in (False, True):
        results = []
        for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
            schema = strawberry.Schema(
                query=ZooQuery, types=types, execution_context_class=cls
            )
            query = "{ animals { __typename name } }"
            result = schema.execute_sync(query) if sync else await schema.execute(query)
            results.append(snapshot(result))
        assert results[0] == results[1]


async def test_enum_serialization_is_cached_per_member() -> None:
    await assert_same("{ items(count: 9) { color } }")


async def test_custom_scalar_returning_null_reports_the_same_error() -> None:
    result = await assert_same("{ items(count: 5) { index nulled } }")
    assert "coerce_output_value" in result


async def test_defer_and_stream_take_the_default_path() -> None:
    config = StrawberryConfig(enable_experimental_incremental_execution=True)
    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = build_schema(cls, [], config)
        result = await schema.execute(
            "{ items(count: 2) { index ... @defer { label greet } } }"
        )
        initial = result.initial_result.formatted
        subsequent = [item.formatted async for item in result.subsequent_results]
        if cls is StrawberryGraphQLCoreExecutionContext:
            expected = (initial, subsequent)
        else:
            assert (initial, subsequent) == expected


async def test_subscriptions_take_the_default_path() -> None:
    @strawberry.type
    class Subscription:
        @strawberry.subscription
        async def ticks(self) -> AsyncGenerator[Item, None]:
            for i in range(3):
                yield make_item(i)

    @strawberry.type
    class SubQuery:
        ping: str = "pong"

    outputs = []
    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = strawberry.Schema(
            query=SubQuery,
            subscription=Subscription,
            execution_context_class=cls,
        )
        generator = await schema.subscribe("subscription { ticks { index label } }")
        outputs.append([event.data async for event in generator])
    assert outputs[0] == outputs[1]


@strawberry.interface
class Shape:
    sides: int

    @classmethod
    def resolve_type(cls, obj: Any, info: Any, *_: Any) -> Any:
        return obj.kind


@strawberry.type
class Square(Shape):
    sides: int


@strawberry.type
class NotAShape:
    sides: int


def make_square(kind: Any) -> Square:
    square = Square(sides=4)
    square.kind = kind  # type: ignore[attr-defined]
    return square


@strawberry.type
class ShapeQuery:
    unrelated: NotAShape | None = None

    @strawberry.field
    def shapes(self) -> list[Shape | None]:
        kinds = ["Square", "NotAShape", "Missing", None, 7, "Square", "Square"]
        return [make_square(kind) for kind in kinds]

    @strawberry.field
    def loose_color(self) -> list[Color | None]:
        return [Color.RED, "red", 3, "nope"]  # type: ignore[list-item]


@pytest.mark.parametrize("sync", [False, True])
async def test_invalid_runtime_types_report_the_same_errors(sync: bool) -> None:
    snapshots = []
    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = strawberry.Schema(
            query=ShapeQuery, types=[Square], execution_context_class=cls
        )
        snapshots.append(await run_query(schema, "{ shapes { sides } }", {}, sync=sync))
    assert snapshots[0] == snapshots[1]
    assert "is not a possible type" in snapshots[1]


@pytest.mark.parametrize("sync", [False, True])
async def test_enum_values_returned_by_resolvers(sync: bool) -> None:
    snapshots = []
    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = strawberry.Schema(
            query=ShapeQuery, types=[Square], execution_context_class=cls
        )
        snapshots.append(await run_query(schema, "{ looseColor }", {}, sync=sync))
    assert snapshots[0] == snapshots[1]


async def test_arguments_are_converted_once_per_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strawberry.execution import field_plan

    calls = []
    original = field_plan.convert_arguments

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(field_plan, "convert_arguments", counting)
    schema = build_schema(FieldPlanExecutionContext, [])
    result = await schema.execute('{ items(count: 6) { greet(name: "q") } }')
    assert result.errors is None
    assert len(calls) == 1


async def test_tracing_subclass_overriding_resolve_is_respected() -> None:
    class Recording(ApolloTracingExtensionSync):
        def resolve(
            self, _next: Any, root: Any, info: Any, *args: Any, **kwargs: Any
        ) -> Any:
            EVENTS.append(f"resolve {info.parent_type.name}.{info.field_name}")
            return super().resolve(_next, root, info, *args, **kwargs)

    result = await assert_same(
        "{ items(count: 2) { index greet } }", extensions=[Recording]
    )
    assert "resolve Item.index" in result


async def test_plan_cache_distinguishes_negative_zero() -> None:
    @strawberry.type
    class Probe:
        @strawberry.field
        def show(self, value: float) -> str:
            return repr(value)

    @strawberry.type
    class FloatQuery:
        @strawberry.field
        def probes(self) -> list[Probe]:
            return [Probe(), Probe()]

    query = "query Q($v: Float!) { probes { show(value: $v) } }"
    outputs = []
    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = strawberry.Schema(
            query=FloatQuery,
            extensions=[ParserCache(maxsize=10)],
            execution_context_class=cls,
        )
        outputs.append(
            [
                (await schema.execute(query, variable_values={"v": v})).data
                for v in (0.0, 0.0, -0.0, 0.0, -0.0)
            ]
        )
    assert outputs[0] == outputs[1]


async def test_mutable_converted_arguments_are_not_shared() -> None:
    @strawberry.type
    class Counter:
        @strawberry.field
        def bump(self, x: strawberry.Maybe[int]) -> int:
            x.value += 1
            return x.value

    @strawberry.type
    class CounterQuery:
        @strawberry.field
        def counters(self) -> list[Counter]:
            return [Counter(), Counter(), Counter()]

    outputs = []
    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        schema = strawberry.Schema(query=CounterQuery, execution_context_class=cls)
        outputs.append((await schema.execute("{ counters { bump(x: 10) } }")).data)
    assert outputs[0] == outputs[1]


def test_concurrent_sync_execution_with_plan_cache() -> None:
    from concurrent.futures import ThreadPoolExecutor

    schema = build_schema(
        FieldPlanExecutionContext, [ParserCache(maxsize=4), ValidationCache(maxsize=4)]
    )
    queries = [
        ("query Q($n: Int!) { items(count: $n) { index greet label } }", {"n": n})
        for n in range(1, 7)
    ]
    expected = {
        n: build_schema(StrawberryGraphQLCoreExecutionContext, [])
        .execute_sync(query, variable_values=variables)
        .data
        for n, (query, variables) in enumerate(queries)
    }

    def run(index: int) -> bool:
        query, variables = queries[index % len(queries)]
        result = schema.execute_sync(query, variable_values=variables)
        return result.errors is None and result.data == expected[index % len(queries)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(run, range(600)))


@strawberry.enum
class Letters(enum.Enum):
    AAA = "aaa"
    BBB = "bbb"


class Chameleon:
    """Non-member value whose enum value changes between executions."""

    def __init__(self) -> None:
        self.value = "aaa"

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return self.value == (other.value if isinstance(other, Chameleon) else other)


CHAMELEON = Chameleon()


@strawberry.type
class LetterHolder:
    @strawberry.field
    def letters(self) -> list[Letters]:
        CHAMELEON.value = "bbb" if CHAMELEON.value == "aaa" else "aaa"
        return [CHAMELEON, CHAMELEON]  # type: ignore[list-item]


@strawberry.type
class LetterQuery:
    @strawberry.field
    def holders(self) -> list[LetterHolder]:
        return [LetterHolder(), LetterHolder()]


async def test_enum_values_built_at_runtime_are_not_confused() -> None:
    outputs = []
    for cls in (StrawberryGraphQLCoreExecutionContext, FieldPlanExecutionContext):
        CHAMELEON.value = "aaa"
        schema = strawberry.Schema(
            query=LetterQuery,
            execution_context_class=cls,
            extensions=[ParserCache(maxsize=4)],
        )
        outputs.append(
            [(await schema.execute("{ holders { letters } }")).data for _ in range(6)]
        )
    assert outputs[0] == outputs[1]
    assert outputs[1][0]["holders"][0] != outputs[1][0]["holders"][1]
