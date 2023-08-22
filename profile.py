"""
@file profile.py
@author Nianchen Deng, Shanghai AI Lab
@brief A profile module for measuring both host-side and device-side performance.

Usage:
1. Create a profiler and enable the profiling: `profiler = Profiler(...).enable()`
    You can specify the number of warmup frames and the number of frames to be recorded when creating the profiler.
    You can also specify a callback function to be called when the profiling is done.
2. Wrap the code to be profiled in one of the following ways:
    a. Use the decorator:
        ```
        @profile
        def func_to_be_profile():
            ...
        ```
    b. Use the context manager:
        ```
        with profile("Node Name"):
            ...
        ```
3. Get the result: `profile_result = profiler.get_result(...)`
    You can specify the name of the node to get the result of a specific node.
    To get the printable report, use `profile_result.get_statistic(metric_keys, ...).get_report(...)`.
    The report can be sorted by a specific metric by specifying the `sort_by` argument for `get_report`.
    To get the raw data with hierarchy information, use `get_raw_data` function instead of `get_report`.

Note:
If you want to instantly get the performance of a piece of code, you can use the `debug_profile` function.
"""
import os
import time
import torch
from itertools import groupby
from typing import Dict, List, TypeVar, Any, Callable, Iterable, Union

__all__ = ["Profiler", "profile", "debug_profile"]


cuda_event_overhead = None
""" The overhead of creating and recording a cuda event, automatically measured before the first frame. """


class ProfileNode:
    name: str
    parent: "ProfileNode"
    children: List["ProfileNode"]
    closed: bool

    @property
    def host_duration(self) -> float:
        if hasattr(self, "_host_duration"):
            return self._host_duration
        if not self.closed:
            raise RuntimeError("Cannot get host duration of an unclosed node")
        self._host_duration = (self._host_end_time - self._host_start_time) * 1000
        return self._host_duration

    @property
    def device_duration(self) -> float:
        if hasattr(self, "_device_duration"):
            return self._device_duration
        if not self.closed:
            raise RuntimeError("Cannot get device duration of an unclosed node")
        self._device_duration = self._device_start_event.elapsed_time(self._device_end_event) \
            - self.cuda_event_overhead_of_children
        self._device_end_event = self._device_start_event = None
        return self._device_duration

    @property
    def cuda_event_overhead_of_children(self) -> float:
        if hasattr(self, "_cuda_event_overhead_of_children"):
            return self._cuda_event_overhead_of_children
        if not self.closed:
            raise RuntimeError("Cannot get cuda event overhead of children of an unclosed node")
        if cuda_event_overhead is None:
            return 0.
        self._cuda_event_overhead_of_children = sum(
            child.cuda_event_overhead_of_children for child in self.children
        ) + cuda_event_overhead * len(self.children) * 2
        return self._cuda_event_overhead_of_children

    def __init__(self, name, parent: "ProfileNode" = None) -> None:
        self.name = name
        self.parent = parent
        self.children = []
        self._device_start_event = torch.cuda.Event(True)
        self._device_end_event = torch.cuda.Event(True)
        self._device_start_event.record()
        self._host_start_time = time.perf_counter()
        self.closed = False

    def add_child(self, name) -> "ProfileNode":
        if self.closed:
            raise RuntimeError("Cannot add child to a closed node")
        child = ProfileNode(name, self)
        self.children.append(child)
        return child

    def close(self) -> "ProfileNode":
        if self.closed:
            raise RuntimeError("The node has been closed")
        self.closed = True
        self._host_end_time = time.perf_counter()
        self._device_end_event.record()
        return self.parent

    def get_result_node(self, parent_path: str = "", cuda_event_overhead: float = 0.) -> "ResultNode":
        if not self.closed:
            raise RuntimeError("Cannot get result of an unclosed node")
        path = f"{parent_path}/{self.name}"
        return ResultNode(
            path, [child.get_result_node(parent_path=path) for child in self.children],
            host_duration=self.host_duration, device_duration=self.device_duration
        )

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__} \"{self.name}\" "
        if self.closed:
            ret += f"[host spent {self.host_duration:2f}ms, device spent {self.device_duration:2f}ms]"
        else:
            ret += "[Not closed]"


class ResultNode:
    path: str
    children: List["ResultNode"]
    data: Dict[str, float]

    @property
    def name(self) -> str:
        return os.path.split(self.path)[-1]

    def __init__(self, path: str, children: List["ResultNode"], **data: float) -> None:
        self.path = path
        self.children = children
        self.data = data

    def value(self, key: str) -> float:
        return self.data[key]
    
    def self_only_value(self, key: str) -> float:
        return self.data[key] - sum([child.value(key) for child in self.children])

    def flatten(self, include_self: bool = True, depth: int = -1) -> List["ResultNode"]:
        result = [self] if include_self else []
        if depth == 0:
            return result
        if depth == 1:
            return result + self.children
        return sum((node.flatten(depth=-1 if depth == -1 else depth - 1) for node in self.children),
                   result)


class ResultNodeCollection:
    """
    A collection of ResultNode objects, representing a group of related profiling results across all profiled frames.
    """

    name: str
    """ (str) The name of the collection, taken from the name of the first ResultNode in the collection. """
    nodes: List[ResultNode]
    """ (list[ResultNode]) The list of ResultNode objects in the collection. """

    def __init__(self, nodes: Iterable[ResultNode]) -> None:
        """
        Initializes a new ResultNodeCollection object.

        Args:
            nodes (Iterable[ResultNode]): An iterable of ResultNode objects to include in the collection.
        """
        self.nodes = list(nodes)
        self.name = self.nodes[0].name if len(self.nodes) > 0 else None

    def get_statistic(self, *keys: str, child_depth: int = -1, return_flatten: bool = False,
                      parent: "StatisticData" = None) -> "StatisticData":
        """
        Calculates and returns a StatisticData object representing the statistics of the collection.

        Args:
            keys (str): The metric keys to include in the statistics, currently supported keys are "host_duration" and "device_duration".
            child_depth (int, optional): The depth of child nodes to include in the statistics. Defaults to -1, which means all children.
            return_flatten (bool, optional): Whether to ignore hierarchy of child nodes in statistics. Defaults to False.
            parent (StatisticData, optional): The parent StatisticData object. Should not be specified when called from outside.

        Returns:
            StatisticData: The calculated statistics for the collection of nodes.
        """
        result = StatisticData(self.name, len(self.nodes) / len(default_profiler.recorded_frames))
        if len(self.nodes) == 0:
            return result
        
        result.data = {}
        for key in keys:
            values = [node.value(key) for node in self.nodes]
            sum_values = sum(values)
            per_frame = sum_values / len(default_profiler.recorded_frames)
            result.data[key] = {
                "min": min(values),
                "max": max(values),
                "sum": per_frame,
                "avg": sum_values / len(values),
                "ratio": per_frame / max(parent.data[key]["sum"], 1e-7) if parent else 1.
            }
        if child_depth == 0:
            return result
        
        children = sum((node.flatten(False, depth=child_depth if return_flatten else 1)
                        for node in self.nodes), [])
        if len(children) == 0:
            return result
        
        children.sort(key=lambda node: node.name)
        child_depth = 0 if return_flatten else child_depth - 1
        result.children = [
            ResultNodeCollection(nodes).get_statistic(*keys, child_depth=child_depth, parent=result)
            for _, nodes in groupby(children, lambda node: node.name)
        ]
        return result


class StatisticData:
    """
    A class representing performance statistics for a node across all profiled frames.
    """
    
    name: str
    """ (str) The name of the node which the statistics are calculated for. """
    n_calls: float
    """ (float) The average number of times the node was invoked in a frame. """
    data: Dict[str, Dict[str, float]]
    """ (dict[str, dict[str, float]]) A dictionary containing statistic data for the node."""
    children: List["StatisticData"]
    """ (list[StatisticData]) A list of StatisticData objects representing the statistics of child nodes."""

    def __init__(self, name: str, n_calls: float) -> None:
        """
        Initializes a StatisticData object with a given name and number of calls.

        Args:
            name (str): The name of the node which the statistics are calculated for.
            n_calls (float): The average number of times the node was invoked in a frame.
        """
        self.name = name
        self.n_calls = n_calls
        self.data = None
        self.children = []

    def get_raw_data(self, sort_by: str = None, decent: str = True, parent_hierarchy: List[str] = []):
        """
        Recursively collects raw data from the statistic data object.

        Args:
            sort_by (str, optional): The metric key to sort the children by. Defaults to None.
            decent (str, optional): Whether to sort in descending order. Defaults to True.
            parent_hierarchy (list[str], optional): The hierarchy of parent nodes. Should not be specified when called from outside.

        Returns:
            list[dict]: A list of dictionaries containing the raw data of this node and its children. Each dictionary contains:
                hierarchy (list[str]): The hierarchy of the node.
                n_calls (float): The average number of times the node was invoked in a frame.
                <metric keys>: The statistics of the node, including "min", "max", "sum", "avg" and "ratio".
        """
        self_hierarchy = parent_hierarchy + [self.name]
        self_data = {
            "hierarchy": self_hierarchy,
            "n_calls": self.n_calls,
            **self.data
        }
        
        if sort_by:
            children = self.children.copy()
            children.sort(key=lambda child: child.data[sort_by]["ratio"], reverse=decent)
        else:
            children = self.children
        return sum((child.get_raw_data(sort_by, decent, self_hierarchy) for child in children),
                    [self_data])
    
    def get_report(self, title: str = None, sort_by: str = None, decent: str = True):
        """
        Returns a printable report of the statistic data.

        Args:
            title (str, optional): The title of the report. Defaults to None.
            sort_by (str, optional): The metric key to sort the children by. Defaults to None.
            decent (str, optional): Whether to sort in descending order. Defaults to True.

        Returns:
            str: The printable performance report.
        """
        s = f"Performance Report" + (f"of {title}" if title else "") + ":\n"
        if self.n_calls == 0:
            s += "No available data.\n"
            return s
        
        flatten_data = self.get_flatten_data(sort_by=sort_by, decent=decent)

        cols = []
        cols_align = []
        cols.append(["Node"] + 
                    ["  " * (len(data["hierarchy"]) - 1) + data["hierarchy"][-1] for data in flatten_data])
        cols_align.append("<")
        cols.append(["# of Calls"] + [f"{data['n_calls']:.1f}" for data in flatten_data])
        cols_align.append(">")
        if "device_duration" in self.data:
            cols.append(["Device Duration"] + 
                        ["{:.2f}ms ({:6.2f}%)".format(data['device_duration']['sum'],
                                                      data['device_duration']['ratio'] * 100)
                         for data in flatten_data])
            cols_align.append(">")
        if "host_duration" in self.data:
            cols.append(["Host Duration"] + 
                        ["{:.2f}ms ({:6.2f}%)".format(data['host_duration']['sum'],
                                                      data['host_duration']['ratio'] * 100)
                         for data in flatten_data])
            cols_align.append(">")
        cols_max_width = [
            max(len(val) for val in col)
            for col in cols
        ]

        # Heading
        cols_zip = zip(*cols)
        heading_items = next(cols_zip)
        s += "  ".join(f"{{:<{col_max_width}s}}" for col_max_width in cols_max_width) \
            .format(*heading_items) + "\n"
        
        # Split line
        s += "=" * (sum(cols_max_width) + 2 * len(cols) - 1) + "\n"

        # Content
        return s + "\n".join(
            "  ".join(
                f"{{:{col_align}{col_max_width}s}}"
                for col_max_width, col_align in zip(cols_max_width, cols_align)
            ).format(*items)
            for items in cols_zip
        )


class Profiler:
    """
    A profiler that records the execution time of code blocks.

    The profiler records the execution time of code blocks as "nodes" in a tree structure.
    The profiler can record a fixed number of frames, or all frames after a warmup period.
    The profiler can also record only nodes up to a certain depth in the tree.
    """

    recorded_frames: List[ResultNode]
    """ (list[ResultNode]) A list of ResultNode objects, each representing a frame of execution. """

    def __init__(self, warmup_frames: int = 0, record_frames: int = 0, record_depth: int = -1,
                 then: Callable[["Profiler"], Any] = None) -> None:
        """
        Initializes a new Profiler object.

        Args:
            warmup_frames (int, optional): the number of frames to discard before recording. Defaults to 0.
            record_frames (int, optional): the number of frames to record after the warmup period. Defaults to 0.
            record_depth (int, optional): the maximum depth of nodes to record. Defaults to -1 (no limit).
            then ((Profiler) -> Any, optional): a function to call after profiling is done. Defaults to None.
        """
        self.recorded_frames = []
        self._enabled = False
        self._root_node = None
        self._current_node = None
        self._warmup_frames = warmup_frames
        self._record_frames = record_frames
        self._frame_counter = 0
        self._then_fn = then
        self._record_depth = record_depth
        self._current_depth = -1
        self._skipped_enters = 0
        self._cuda_event_overhead = 0.

    def enable(self) -> "Profiler":
        """
        Enables the profiler and set as default.

        Returns:
            self, for method chaining.
        """
        global default_profiler
        if default_profiler is not None and default_profiler._enabled:
            default_profiler._enabled = False
        default_profiler = self
        default_profiler._enabled = True
        return self
    
    def disable(self) -> "Profiler":
        """
        Disables the profiler.

        Returns:
            (self) for method chaining.
        """
        global default_profiler
        default_profiler = None
        self._enabled = False
        return self
    
    def enter_node(self, name: str):
        """
        Enters a new node.

        Args:
            name (str): the name of the node.
        """
        if not self._enabled \
            or (self._record_depth >= 0 and self._current_depth >= self._record_depth):
            self._skipped_enters += 1
            return
        if self._current_node is None:
            global cuda_event_overhead
            if self._frame_counter == self._warmup_frames:
                cuda_event_overhead = self._measure_cuda_event_overhead()
            self._root_node = self._current_node = ProfileNode(name)
        else:
            self._current_node = self._current_node.add_child(name)
        self._current_depth += 1

    def leave_node(self):
        """
        Leaves the current node.
        """
        if self._skipped_enters > 0:
            self._skipped_enters -= 1
            return
        self._current_node = self._current_node.close()
        self._current_depth -= 1
        if self._current_node is None:
            # torch.cuda.synchronize()
            self._root_node._device_end_event.synchronize()
            if self._frame_counter >= self._warmup_frames:
                self.recorded_frames.append(
                    self._root_node.get_result_node(cuda_event_overhead=self._cuda_event_overhead))
            self._root_node = None
            self._frame_counter += 1
            if self._record_frames > 0 \
                and self._frame_counter >= self._warmup_frames + self._record_frames:
                self._enabled = False
                if self._then_fn is not None:
                    self._then_fn(self)

    def get_result(self, name: str = None) -> ResultNodeCollection:
        """
        Collect the result nodes across the recorded frames.

        Args:
            name (str): if specified, returns only nodes with this name and their children. Defaults to None.

        Returns:
            (ResultNodeCollection) a collection of result nodes.
        """
        if name is None:
            return ResultNodeCollection(self.recorded_frames)
        else:
            all_nodes = sum((node.flatten() for node in self.recorded_frames), [])
            return ResultNodeCollection([node for node in all_nodes if node.name == name])

    def _measure_cuda_event_overhead(self, n_samples: int = 1000) -> float:
        """
        Measures the overhead of CUDA event's create and record operation.

        Args:
            n_samples (int): the number of samples to take. Defaults to 1000.

        Returns:
            (float) the overhead in milliseconds.
        """
        torch.cuda.synchronize()
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(n_samples):
            torch.cuda.Event(True).record()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / n_samples


default_profiler: Profiler = None


class _ProfileWrap:
    def __init__(self, fn: Callable = None, name: str = None) -> None:
        self.fn = fn
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.fn == None and len(args) == 1 and isinstance(args[0], Callable):
            self.fn = args[0]
            return lambda *args, **kwargs: self(*args, **kwargs)
        self.__enter__()
        ret = self.fn(*args, **kwargs)
        self.__exit__()
        return ret

    def __enter__(self):
        if default_profiler is not None:
            default_profiler.enter_node(self.name or self.fn.__qualname__)
        return self

    def __exit__(self, *args, **kwargs):
        if default_profiler is not None:
            default_profiler.leave_node()


class _DebugProfileWrap:
    def __init__(self, fn: Callable = None, name: str = None) -> None:
        self.fn = fn
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.fn == None and len(args) == 1 and isinstance(args[0], Callable):
            self.fn = args[0]
            return lambda *args, **kwargs: self(*args, **kwargs)
        self.__enter__()
        ret = self.fn(*args, **kwargs)
        self.__exit__()
        return ret

    def __enter__(self):
        self.node = ProfileNode(self.name)
        return self

    def __exit__(self, *args, **kwargs):
        self.node.close()
        torch.cuda.synchronize()
        print(f"Node {self.name}: host duration {self.node.host_duration:.1f}ms, "
              f"device duration {self.node.device_duration:.1f}ms")


FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)

def profile(arg: Union[str, F]) -> Union[_ProfileWrap, F]:
    """
    Profile a function or a block of codes.

    This function can be used as a decorator or can be directly called to create context manager.

    Args:
        arg (str | function): Either the name of the profile or the function to be profiled.
    """
    if isinstance(arg, str):
        return _ProfileWrap(name=arg)
    else:
        return lambda *args, **kwargs: _ProfileWrap(fn=arg)(*args, **kwargs)


def debug_profile(arg: Union[str, Callable]):
    """
    Profile a function or a block of codes and immediately print the result.

    This function is used for debug, and will not add node to the default profiler.

    Args:
        arg (str | function): Either the name of the profile or the function to be profiled.
    """
    if isinstance(arg, str):
        return _DebugProfileWrap(name=arg)
    else:
        return lambda *args, **kwargs: _DebugProfileWrap(fn=arg)(*args, **kwargs)