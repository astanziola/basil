from basil.utils import camel_to_snake_case


class Benchmark:

    @property
    def name(self):
        # Transform class name into snake case
        return camel_to_snake_case(self.__class__.__name__)

    def validate_outputs(self, outputs):
        pass
