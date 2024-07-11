from metaflow import FlowSpec, step, nvidia
from metaflow.profilers import gpu
from custom_deco import pip

class HelloDGXCloud(FlowSpec):

    @pip(packages={'torch': '2.3.1'})
    @nvidia
    @step
    def start(self):
        import torch
        print(
            f"Hello from {torch.__name__} version {torch.__version__}!"
        )
        if torch.cuda.is_available():
            print(
                f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB)"
            )
        else:
            print("No GPU available.")
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    HelloDGXCloud()
    
