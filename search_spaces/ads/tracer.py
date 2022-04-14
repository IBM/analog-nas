import torch
import torch.fx as fx
import torchvision.models as models 
from typing import Any, Callable, Dict, Optional, Tuple


class ModulePathTracer(torch.fx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """
    current_module_qualified_name : str = ''
    node_to_originating_module : Dict[torch.fx.Node, str] = {}
    modules= []
    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module
        """
        old_qualname = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(m)
            return super().call_module(m, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`.
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = args
        return proxy


def main():
    model = models.resnet50(pretrained=True)

    # Instantiate our ModulePathTracer and use that to trace our ResNet18
    tracer = ModulePathTracer()
    traced_network = tracer.trace(model)
    i = 0
    for node in traced_network.nodes:
        module_qualname = tracer.node_to_originating_module.get(node)
        print("node {}:".format(i), node, "originate:", module_qualname)
        i += 1

    
if __name__ == "__main__":
    main()
        


