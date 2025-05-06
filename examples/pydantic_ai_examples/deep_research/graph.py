from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, overload

from .nodes import Node, NodeId, TypeUnion


class Routing[T]:
    """This is an auxiliary class that is purposely not a dataclass, and should not be instantiated.

    It should only be used for its `__class_getitem__` method.
    """

    _force_invariant: Callable[[T], T]


@dataclass
class CallNode[StateT, InputT, OutputT](Node[StateT, InputT, OutputT]):
    id: NodeId
    call: Callable[[StateT, InputT], Awaitable[OutputT]]

    async def run(self, state: StateT, inputs: InputT) -> OutputT:
        return await self.call(state, inputs)


@dataclass
class Interruption[StopT, ResumeT]:
    value: StopT
    next_node: Node[Any, ResumeT, Any]


class EmptyNodeFunction[OutputT](Protocol):
    def __call__(self) -> OutputT:
        raise NotImplementedError


class StateNodeFunction[StateT, OutputT](Protocol):
    def __call__(self, state: StateT) -> OutputT:
        raise NotImplementedError


class InputNodeFunction[InputT, OutputT](Protocol):
    def __call__(self, inputs: InputT) -> OutputT:
        raise NotImplementedError


class FullNodeFunction[StateT, InputT, OutputT](Protocol):
    def __call__(self, state: StateT, inputs: InputT) -> OutputT:
        raise NotImplementedError


@overload
def node[OutputT](fn: EmptyNodeFunction[OutputT]) -> Node[Any, None, OutputT]: ...
@overload
def node[InputT, OutputT](
    fn: InputNodeFunction[InputT, OutputT],
) -> Node[Any, InputT, OutputT]: ...
@overload
def node[StateT, OutputT](
    fn: StateNodeFunction[StateT, OutputT],
) -> Node[StateT, None, OutputT]: ...
@overload
def node[StateT, InputT, OutputT](
    fn: FullNodeFunction[StateT, InputT, OutputT],
) -> Node[StateT, InputT, OutputT]: ...


def node(fn: Callable[..., Any]) -> Node[Any, Any, Any]:
    signature = inspect.signature(fn)
    signature_error = "Function may only make use of parameters 'state' and 'inputs'"
    node_id = NodeId(fn.__name__)
    if 'state' in signature.parameters and 'inputs' in signature.parameters:
        assert len(signature.parameters) == 2, signature_error
        return CallNode(id=node_id, call=fn)
    elif 'state' in signature.parameters:
        assert len(signature.parameters) == 1, signature_error
        return CallNode(id=node_id, call=lambda state, inputs: fn(state))
    elif 'state' in signature.parameters:
        assert len(signature.parameters) == 1, signature_error
        return CallNode(id=node_id, call=lambda state, inputs: fn(inputs))
    else:
        assert len(signature.parameters) == 0, signature_error
        return CallNode(id=node_id, call=lambda state, inputs: fn())


type Router[StateT, GraphOutputT, SourceInputT, SourceOutputT] = Callable[
    [HandleMaker[StateT, GraphOutputT, SourceInputT]], type[Routing[SourceOutputT]]
]


@dataclass
class GraphBuilder[StateT, InputT, OutputT]:
    # TODO: Should get the following values from __class_getitem__ somehow;
    #   this would make it possible to use typeforms without type errors
    state_type: type[StateT] = field(init=False)
    input_type: type[InputT] = field(init=False)
    output_type: type[OutputT] = field(init=False)

    _start_at: Router[StateT, OutputT, InputT, InputT] | Node[StateT, InputT, Any]
    _simple_edges: list[
        tuple[
            Node[StateT, Any, Any],
            TransformFunction[StateT, Any, Any, Any] | None,
            Node[StateT, Any, Any],
        ]
    ] = field(init=False, default_factory=list)
    _routed_edges: list[
        tuple[Node[StateT, Any, Any], Router[StateT, OutputT, Any, Any]]
    ] = field(init=False, default_factory=list)

    def edge[T](
        self,
        *,
        source: Node[StateT, Any, T],
        transform: TransformFunction[StateT, Any, Any, T] | None = None,
        destination: Node[StateT, T, Any],
    ):
        self._simple_edges.append((source, transform, destination))

    def edges[SourceInputT, SourceOutputT](
        self,
        source: Node[StateT, SourceInputT, SourceOutputT],
        routing: Router[StateT, OutputT, SourceInputT, SourceOutputT],
    ):
        self._routed_edges.append((source, routing))

    def build(self) -> Graph[StateT, InputT, OutputT]:
        # TODO: Build nodes from edges/decisions
        nodes: dict[NodeId, Node[StateT, Any, Any]] = {}
        assert self._start_at is not None, (
            'You must call `GraphBuilder.start_at` before building the graph.'
        )
        return Graph[StateT, InputT, OutputT](
            nodes=nodes,
            start_at=self._start_at,
            edges=[(e[0].id, e[1], e[2].id) for e in self._simple_edges],
            routed_edges=[(d[0].id, d[1]) for d in self._routed_edges],
        )

    def _check_output(self, output: OutputT) -> None:
        raise RuntimeError(
            'This method is only included for type-checking purposes and should not be called directly.'
        )


@dataclass
class Graph[StateT, InputT, OutputT]:
    nodes: dict[NodeId, Node[StateT, Any, Any]]

    # TODO: May need to tweak the following to actually work at runtime...
    start_at: Router[StateT, OutputT, InputT, InputT] | Node[StateT, InputT, Any]
    edges: list[tuple[NodeId, Any, NodeId]]
    routed_edges: list[tuple[NodeId, Router[StateT, OutputT, Any, Any]]]

    @staticmethod
    def builder[S, I, O](
        state_type: type[S],
        input_type: type[I],
        output_type: type[TypeUnion[O]] | type[O],
        start_at: Node[S, I, Any] | Router[S, O, I, I],
    ) -> GraphBuilder[S, I, O]:
        raise NotImplementedError

    def run(self, state: StateT, inputs: InputT) -> OutputT:
        raise NotImplementedError

    def resume[NodeInputT](
        self,
        state: StateT,
        node: Node[StateT, NodeInputT, Any],
        node_inputs: NodeInputT,
    ) -> OutputT:
        raise NotImplementedError


class HandleMaker[GraphStateT, GraphOutputT, HandleInputT](Protocol):
    def __call__[T](
        self, type_: type[T], is_instance: Callable[[Any], bool] | None = None
    ) -> Handle[T, GraphStateT, GraphOutputT, HandleInputT, T]:
        if is_instance is None:
            is_instance = lambda v: isinstance(v, type_)
        return Handle(type_, is_instance)


class _TransformState[StateT, T](Protocol):
    def __call__(self, state: StateT) -> T:
        raise NotImplementedError


class _TransformInput[InputT, T](Protocol):
    def __call__(self, inputs: InputT) -> T:
        raise NotImplementedError


class _TransformOutput[OutputT, T](Protocol):
    def __call__(self, output: OutputT) -> T:
        raise NotImplementedError


class _TransformStateInput[StateT, InputT, T](Protocol):
    def __call__(self, state: StateT, inputs: InputT) -> T:
        raise NotImplementedError


class _TransformStateOutput[StateT, OutputT, T](Protocol):
    def __call__(self, state: StateT, output: OutputT) -> T:
        raise NotImplementedError


class _TransformInputOutput[InputT, OutputT, T](Protocol):
    def __call__(self, inputs: InputT, output: OutputT) -> T:
        raise NotImplementedError


class _TransformStateInputOutput[StateT, InputT, OutputT, T](Protocol):
    def __call__(self, state: StateT, inputs: InputT, output: OutputT) -> T:
        raise NotImplementedError


type TransformFunction[StateT, SourceInputT, SourceOutputT, DestinationInputT] = (
    _TransformState[StateT, DestinationInputT]
    | _TransformInput[SourceInputT, DestinationInputT]
    | _TransformOutput[SourceOutputT, DestinationInputT]
    | _TransformStateInput[StateT, SourceInputT, DestinationInputT]
    | _TransformStateOutput[StateT, SourceOutputT, DestinationInputT]
    | _TransformInputOutput[SourceInputT, SourceOutputT, DestinationInputT]
    | _TransformStateInputOutput[StateT, SourceInputT, SourceOutputT, DestinationInputT]
)


@dataclass
class Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, HandleOutputT]:
    _source_type: type[SourceT]
    _is_instance: Callable[[Any], bool]
    _transforms: tuple[TransformFunction[GraphStateT, HandleInputT, Any, Any], ...] = (
        field(default=())
    )

    _end: bool = field(init=False, default=False)

    # Note: _route_to must use `Any` instead of `HandleOutputT` in the first argument to keep this type contravariant in
    # HandleOutputT. I _believe_ this is safe because instances of this type should never get mutated after this is set.
    _route_to: Node[GraphStateT, Any, Any] | None = field(init=False, default=None)

    def end(
        self: Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, GraphOutputT],
    ) -> type[SourceT]:
        self._end = True
        return self._source_type

    def route_to(self, node: Node[GraphStateT, HandleOutputT, Any]) -> type[SourceT]:
        self._route_to = node
        return self._source_type

    @overload
    def transform[T](
        self, call: _TransformState[GraphStateT, T]
    ) -> Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, T]: ...

    @overload
    def transform[T](
        self, call: _TransformInput[HandleInputT, T]
    ) -> Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, T]: ...

    @overload
    def transform[T](
        self, call: _TransformOutput[HandleOutputT, T]
    ) -> Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, T]: ...

    @overload
    def transform[T](
        self, call: _TransformStateInput[GraphStateT, HandleInputT, T]
    ) -> Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, T]: ...

    @overload
    def transform[T](
        self, call: _TransformStateOutput[GraphStateT, HandleOutputT, T]
    ) -> Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, T]: ...

    @overload
    def transform[T](
        self, call: _TransformInputOutput[HandleInputT, HandleOutputT, T]
    ) -> Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, T]: ...

    @overload
    def transform[T](
        self,
        call: _TransformStateInputOutput[GraphStateT, HandleInputT, HandleOutputT, T],
    ) -> Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, T]: ...

    def transform[T](
        self, call: TransformFunction[Any, Any, Any, T]
    ) -> Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, T]:
        new_transforms = self._transforms + (call,)
        return Handle(self._source_type, self._is_instance, new_transforms)
