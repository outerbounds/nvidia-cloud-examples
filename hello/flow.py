from metaflow import FlowSpec, step, nvidia, pypi, conda, kubernetes, card, current
from metaflow_extensions.outerbounds.profilers import gpu_profile
from metaflow.cards import ProgressBar

class HelloDGXCloud(FlowSpec):

    @card(type="blank", refresh_interval=1)
    @gpu_profile(interval=1)
    @pypi(python='3.12', packages={'torch': '2.3.1'})
    @nvidia
    @step
    def start(self):
        import os
        print(os.system("nvidia-smi"))
        import torch # pylint: disable=import-error
        print(
            f"Hello from {torch.__name__} version {torch.__version__}!"
        )
        if torch.cuda.is_available():
            self.torch_device_name = torch.cuda.get_device_name(0)
            self.torch_device_mem = torch.cuda.get_device_properties(0).total_memory
            p = ProgressBar(max=10, label="Iteration")
            current.card.append(p)
            current.card.refresh()
            for i in range(10):
                p.update(i+1)
                current.card.refresh()
                tensor_size = (i * 1000, 1_000_000) 
                data = torch.randn(tensor_size).to("cuda")
                data = data * 1.7777
                data = data + 1.3333
                data = torch.sqrt(data)
                del data
                torch.cuda.empty_cache()
        else:
            print("No GPU available.")
        self.next(self.end)

    @step
    def end(self):
        # passing data from @nvidia runtime to other steps
        print(f"Device name: {self.torch_device_name}")
        print(f"Device memory: {self.torch_device_mem / 1024 ** 3:.1f}GB")
        print("Flow finished.")

if __name__ == '__main__':
    HelloDGXCloud()
    
