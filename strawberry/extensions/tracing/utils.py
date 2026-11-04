from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strawberry.extensions.utils import is_introspection_field
from strawberry.resolvers import is_default_resolver

if TYPE_CHECKING:
    from collections.abc import Callable

    from graphql import GraphQLField, GraphQLResolveInfo


def should_skip_tracing(resolver: Callable[..., Any], info: GraphQLResolveInfo) -> bool:
    if info.field_name not in info.parent_type.fields:
        return True
    resolver = info.parent_type.fields[info.field_name].resolve
    return (
        is_introspection_field(info)
        or is_default_resolver(resolver)
        or resolver is None
    )


def traces_field(field_def: GraphQLField) -> bool:
    resolver = field_def.resolve
    return not (resolver is None or is_default_resolver(resolver))


__all__ = ["should_skip_tracing", "traces_field"]
