import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Create 2 virtual GPUs with 1GB memory each
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [
                tf.config.LogicalDeviceConfiguration(memory_limit=1024),
                tf.config.LogicalDeviceConfiguration(memory_limit=1024),
            ],
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs
        # have been initialized
        print(e)
