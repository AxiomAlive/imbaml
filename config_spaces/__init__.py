
class MLModelGenerator:
    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):

        class_attributes = {}
        for cls_ in [cls, cls.__base__]:
            class_attributes.update({k: v for k, v in vars(cls_).items() if not k.startswith('_') and type(v) != classmethod})

        return class_attributes
