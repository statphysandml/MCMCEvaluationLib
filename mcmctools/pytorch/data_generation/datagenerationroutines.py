
def data_generator_factory(data_generator_name="ConfigDataGenerator"):
    if data_generator_name == "BatchConfigDataGenerator":
        from mcmctools.pytorch.data_generation.batchconfigdatagenerator import BatchConfigDataGenerator
        return BatchConfigDataGenerator
    elif data_generator_name == "BatchGraphDataGenerator":
        from mcmctools.pytorch.data_generation.batchgraphdatagenerator import BatchGraphDataGenerator
        return BatchGraphDataGenerator
    else:
        from mcmctools.pytorch.data_generation.configdatagenerator import ConfigDataGenerator
        return ConfigDataGenerator
