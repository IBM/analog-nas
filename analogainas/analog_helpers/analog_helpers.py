# TORCH IMPORTS
# AIHWKIT IMPORTS
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightClipType
from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.optim import AnalogSGD

from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.simulator.parameters.enums import BoundManagementType


def create_noise_model():
    # Returns a noise model for the inference.
    # Would have preferred to use CustomDriftPCMLikeNoiseModel but it is not available in the current environment supported in this repo.
    # g_min, g_max = 0.0, 25.
    # custom_drift_model = dict(g_lst=[g_min, 10., g_max],
    #                           nu_mean_lst=[0.08, 0.05, 0.03],
    #                           nu_std_lst=[0.03, 0.02, 0.01])
    #
    # noise_model = CustomDriftPCMLikeNoiseModel(custom_drift_model,
    #                                            prog_noise_scale=0.0,   # turn off to show drift only
    #                                            read_noise_scale=0.0,   # turn off to show drift only
    #                                            drift_scale=1.0,
    #                                            g_converter=SinglePairConductanceConverter(g_min=g_min,
    #                                                                                       g_max=g_max),
    #                                            )
    noise_model = PCMLikeNoiseModel()
    return noise_model

def create_rpu_config(g_max=25,
                      tile_size=256,
                      dac_res=256,
                      adc_res=256,
                      noise_std=5.0):
    # Returns an RPU configuration for the inference based on the given parameters.
    # Implementation from AIHWKit
    rpu_config = InferenceRPUConfig()

    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.pdrop = 0  # Drop connect.

    rpu_config.modifier.std_dev = noise_std

    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 0.4
    rpu_config.mapping.weight_scaling_omega = True
    rpu_config.mapping.max_input_size = tile_size
    rpu_config.mapping.max_output_size = 255

    rpu_config.mapping.learn_out_scaling_alpha = True

    rpu_config.forward = PresetIOParameters()
    rpu_config.forward.inp_res = 1/dac_res  # 8-bit DAC discretization.
    rpu_config.forward.out_res = 1/adc_res  # 8-bit ADC discretization.
    rpu_config.forward.bound_management = BoundManagementType.NONE

    # Inference noise model.
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max)

    # drift compensation
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config