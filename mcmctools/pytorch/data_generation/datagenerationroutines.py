
def data_generator_factory(data_generator_name="ConfigDataGenerator"):
    if data_generator_name == "BatchConfigDataGenerator":
        from mcmctools.pytorch.data_generation.batchconfigdatagenerator import BatchConfigDataGenerator
        return BatchConfigDataGenerator
    else:
        from mcmctools.pytorch.data_generation.configdatagenerator import ConfigDataGenerator
        return ConfigDataGenerator
