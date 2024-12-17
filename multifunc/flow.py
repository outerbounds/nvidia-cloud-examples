from metaflow import FlowSpec, step, nvidia, current, card, schedule
from metaflow.profilers import gpu_profile
from metaflow.cards import Table

@schedule(daily=True)
class TestDGXModes(FlowSpec):

    headers = ["Step name", "Actual N GPU", "Expected N GPU", "Expected GPU Type", "Actual GPU Type"]

    @step
    def start(self):
        self.next(self.default, self.single_l40)

    def f(self):
        import subprocess
        subprocess.run(["pip", "install", "torch"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        import torch # pylint: disable=import-error
        n = torch.cuda.device_count()
        gpu_type = torch.cuda.get_device_name(0)
        return n, gpu_type

    @gpu_profile(interval=1)
    @nvidia
    @step
    def default(self):
        n_gpu, gpu_type = self.f()

        if n_gpu != 1:
            print(f"Expected 1 GPU, got {n_gpu}.")
        if "H100" not in gpu_type:
            print(f"Expected H100, got {gpu_type}.")
        self.data = [current.step_name, n_gpu,  1,  "H100", gpu_type]

        self.next(self.join)

    @gpu_profile(interval=1)
    @nvidia(gpu_type="L40") # default is gpu=1
    @step
    def single_l40(self):
        n_gpu, gpu_type = self.f()

        if n_gpu != 1:
            print(f"Expected 1 GPU, got {n_gpu}.")
        if "L40" not in gpu_type:
            print(f"Expected L40, got {gpu_type}.")

        self.data = [current.step_name, n_gpu,  1,  "L40", gpu_type]
        self.next(self.join)

    @step
    def join(self, inputs):
        self.prev_data = [i.data for i in inputs]
        self.next(self.tween)

    @step
    def tween(self):
        self.next(self.single_h100, self.multi_h100)

    @gpu_profile(interval=1)
    @nvidia(gpu=1, gpu_type="H100")
    @step
    def single_h100(self):
        n_gpu, gpu_type = self.f()

        if n_gpu != 1:
            print(f"Expected 1 GPU, got {n_gpu}.")
        if "H100" not in gpu_type:
            print(f"Expected H100, got {gpu_type}.")
        
        self.data = [current.step_name, n_gpu,  1,  "H100", gpu_type]
        self.next(self.join2)

    @gpu_profile(interval=1)
    @nvidia(gpu=4, gpu_type="H100")
    @step
    def multi_h100(self):
        n_gpu, gpu_type = self.f()

        if n_gpu != 4:
            print(f"Expected 4 GPUs, got {n_gpu}.")
        if "H100" not in gpu_type:
            print(f"Expected H100, got {gpu_type}.")
        
        self.data = [current.step_name, n_gpu,  4,  "H100", gpu_type]
        self.next(self.join2)

    @step
    def join2(self, inputs):
        self.data = []
        for i in inputs:
            self.data.append(i.data)
            for j in i.prev_data:
                self.data.append(j)
        self.next(self.end)

    @card(type='blank', id='data')
    @step
    def end(self):
        current.card['data'].append(Table(headers=self.headers, data=self.data))

if __name__ == '__main__':
    TestDGXModes()