import tensorflow as tf

# physical_devices = tf.config.list_physical_devices("CPU")
# assert len(physical_devices) == 1, "No CPUs found"
# # Specify 2 virtual CPUs. Note currently memory limit is not supported.
# try:
#     tf.config.set_logical_device_configuration(
#         physical_devices[0],
#         [
#             tf.config.LogicalDeviceConfiguration(),
#             tf.config.LogicalDeviceConfiguration(),
#             tf.config.LogicalDeviceConfiguration(),
#             tf.config.LogicalDeviceConfiguration(),
#         ],
#     )
#     logical_devices = tf.config.list_logical_devices("CPU")
#     print(len(logical_devices))
#     assert len(logical_devices) == 4

# except:
#     # Cannot modify logical devices once initialized.
#     print("Yoooo")


# physical_devices = tf.config.list_physical_devices("GPU")
# try:
#     tf.config.set_logical_device_configuration(
#         physical_devices[0],
#         [
#             tf.config.LogicalDeviceConfiguration(memory_limit=100),
#             tf.config.LogicalDeviceConfiguration(memory_limit=100),
#         ],
#     )

#     logical_devices = tf.config.list_logical_devices("GPU")
#     print(len(logical_devices))
#     assert len(logical_devices) == len(physical_devices) + 1

# except:
#     # Invalid device or cannot modify logical devices once initialized.
#     print("GPU Yoooo")


# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.list_physical_devices("GPU")
# if not gpus:
#     raise ValueError("At least one GPU required for this test!")
# if len(gpus) == 1:
#     # Create two virtual GPUs for this test:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [
#             tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#             tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#         ],
#     )
#     logical_gpus = tf.config.list_logical_devices("GPU")
#     print(f"{len(gpus)} physical GPUs, split into {len(logical_gpus)} logical GPUs")
#     print(logical_gpus)

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
        # Virtual devices must be set before GPUs have been initialized
        print(e)
