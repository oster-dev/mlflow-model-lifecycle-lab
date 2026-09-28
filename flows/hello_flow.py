from metaflow import FlowSpec, step


class HelloFlow(FlowSpec):
    @step
    def start(self):
        print("Metaflow says: Hi!")
        self.next(self.end)

    @step
    def end(self):
        print("Flow finished.")


if __name__ == "__main__":
    HelloFlow()