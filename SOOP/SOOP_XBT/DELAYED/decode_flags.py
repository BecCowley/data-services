""" Use this snippet of code to decode binary flag codes into meaningful values
    Have to pass in flag labels or names as type list of strings.
    And i has to be an unsigned integer
    """


def convert(i, labels):
    output = ''
    pos = 0
    while i:
        if i & 1:
            output += labels[pos]
        pos += 1
        i >>= 1
    return output


# test it:
val = 8388641
labels = ["scientific_qc_applied","wire_stretch","leakage","electrical_interference_interpolated","high_frequency_noise_filtered","repeat_profile","temperature_inversion_confirmed","temperature_inversion_unconfirmed","temperature_offset","temperature_eddy_or_front","temperature_steps_or_structure_confirmed","temperature_steps_or_structure_unconfirmed","depth_offset","constant_temperature","time_error_corrected","latitude_error_corrected","longitude_error_corrected","probe_type_changed","spike_interpolated","fine_structure","insulation_penetration_interpolated","nub_inversion","hit_bottom","premature_launch","surface_temperature_anomaly","surface_offset","temperature_anomaly","temperature_difference_at_depth","unique_id_changed","bowing_BathySystems","cusping_BathySystems_leakage","sippicanMK_timing_delay_driver_error","BathySystem_software_fault_modulo_10_spikes_filtered","protecho_systems_leakage_fault","sippicanMK9_sticking_bit_19_point_filtered","depth_corrected_multiplied_by_10","depth_corrected_update_fall_rate_equation","bathy_data_low_resolution"]
print(convert(val, labels))

