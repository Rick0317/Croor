default_final_bond_dim = 1
default_sweep_schedule_bond_dims = [default_final_bond_dim] * 4 + [
    default_final_bond_dim + 3
] * 4
default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0] # [0] * 10
default_sweep_schedule_davidson_threshold = [1e-20] * 8

init_state_bond_dimension = 10
max_num_sweeps = 200
energy_convergence_threshold = 1e-8
sweep_schedule_bond_dims = default_sweep_schedule_bond_dims
sweep_schedule_noise = default_sweep_schedule_noise
sweep_schedule_davidson_threshold = (
    default_sweep_schedule_davidson_threshold
)
nuc_rep_energy = 0

def get_dmrg_config(num_orbitals, num_electrons, num_unpaired_electrons,
                   multiplicity):
    nuc_rep_energy = 0
    dmrg_param = {"factor_half_convention": True,
                  "symmetry_type": "SZ",
                  "num_threads": 1,
                  "n_mkl_threads": 1,
                  "num_orbitals": num_orbitals,
                  "num_spin_orbitals": 2 * num_orbitals,
                  "num_electrons": num_electrons,
                  "two_S": num_unpaired_electrons,
                  "two_Sz": int((multiplicity - 1) / 2),
                  "orb_sym": None,
                  "temp_dir": "./temp",
                  "stack_mem": 1073741824,
                  "restart_dir": "./restart",
                  "core_energy": nuc_rep_energy,
                  "reordering_method": "none",
                  "init_state_seed": 64241,  # 0 means random seed
                  "initial_mps_method": "random",
                  "init_state_bond_dimension": init_state_bond_dimension,
                  "occupancy_hint": None,
                  "full_fci_space_bool": True,
                  "init_state_direct_two_site_construction_bool": False,
                  "max_num_sweeps": max_num_sweeps,
                  "energy_convergence_threshold":energy_convergence_threshold,
                  "sweep_schedule_bond_dims":sweep_schedule_bond_dims,
                  "sweep_schedule_noise": sweep_schedule_noise,
                  "sweep_schedule_davidson_threshold":sweep_schedule_davidson_threshold,
                  "davidson_type": None,  # Default is None, for "Normal"
                  "eigenvalue_cutoff": 1e-120,
                  "davidson_max_iterations": 4000,  # Default is 4000
                  "davidson_max_krylov_subspace_size": 50,  # Default is 50
                  "lowmem_noise_bool": False,
                  "sweep_start": 0,  # Default is 0, where to start sweep
                  "initial_sweep_direction": None,
                  "stack_mem_ratio": 0.4,  # Default is 0.4
                  }
    return dmrg_param

def get_custom_bd_dmrg_config(num_orbitals, num_electrons, num_unpaired_electrons,
                   multiplicity, bd):
    default_final_bond_dim = bd
    default_sweep_schedule_bond_dims = [default_final_bond_dim] * 8
    default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0]  # [0] * 10
    default_sweep_schedule_davidson_threshold = [1e-20] * 8

    init_state_bond_dimension = bd
    max_num_sweeps = 200
    energy_convergence_threshold = 1e-8
    sweep_schedule_bond_dims = default_sweep_schedule_bond_dims
    sweep_schedule_noise = default_sweep_schedule_noise
    sweep_schedule_davidson_threshold = (
        default_sweep_schedule_davidson_threshold
    )
    nuc_rep_energy = 0
    dmrg_param = {"factor_half_convention": True,
                  "symmetry_type": "SZ",
                  "num_threads": 1,
                  "n_mkl_threads": 1,
                  "num_orbitals": num_orbitals,
                  "num_spin_orbitals": 2 * num_orbitals,
                  "num_electrons": num_electrons,
                  "two_S": num_unpaired_electrons,
                  "two_Sz": int((multiplicity - 1) / 2),
                  "orb_sym": None,
                  "temp_dir": "./temp",
                  "stack_mem": 1073741824,
                  "restart_dir": "./restart",
                  "core_energy": nuc_rep_energy,
                  "reordering_method": "none",
                  "init_state_seed": 64241,  # 0 means random seed
                  "initial_mps_method": "random",
                  "init_state_bond_dimension": init_state_bond_dimension,
                  "occupancy_hint": None,
                  "full_fci_space_bool": True,
                  "init_state_direct_two_site_construction_bool": False,
                  "max_num_sweeps": max_num_sweeps,
                  "energy_convergence_threshold":energy_convergence_threshold,
                  "sweep_schedule_bond_dims":sweep_schedule_bond_dims,
                  "sweep_schedule_noise": sweep_schedule_noise,
                  "sweep_schedule_davidson_threshold":sweep_schedule_davidson_threshold,
                  "davidson_type": None,  # Default is None, for "Normal"
                  "eigenvalue_cutoff": 1e-120,
                  "davidson_max_iterations": 4000,  # Default is 4000
                  "davidson_max_krylov_subspace_size": 50,  # Default is 50
                  "lowmem_noise_bool": False,
                  "sweep_start": 0,  # Default is 0, where to start sweep
                  "initial_sweep_direction": None,
                  "stack_mem_ratio": 0.4,  # Default is 0.4
                  }
    return dmrg_param


def get_dmrg_config_custom(num_orbitals, num_electrons, num_unpaired_electrons,
                   multiplicity, initial_state_bond_dimension):
    default_sweep_schedule_bond_dims = [initial_state_bond_dimension] * 4 + [
        initial_state_bond_dimension + 3
    ] * 4
    nuc_rep_energy = 0
    dmrg_param = {"factor_half_convention": True,
                  "symmetry_type": "SZ",
                  "num_threads": 1,
                  "n_mkl_threads": 1,
                  "num_orbitals": num_orbitals,
                  "num_spin_orbitals": 2 * num_orbitals,
                  "num_electrons": num_electrons,
                  "two_S": num_unpaired_electrons,
                  "two_Sz": int((multiplicity - 1) / 2),
                  "orb_sym": None,
                  "temp_dir": "./temp",
                  "stack_mem": 1073741824,
                  "restart_dir": "./restart",
                  "core_energy": nuc_rep_energy,
                  "reordering_method": "none",
                  "init_state_seed": 64241,  # 0 means random seed
                  "initial_mps_method": "random",
                  "init_state_bond_dimension": initial_state_bond_dimension,
                  "occupancy_hint": None,
                  "full_fci_space_bool": True,
                  "init_state_direct_two_site_construction_bool": False,
                  "max_num_sweeps": max_num_sweeps,
                  "energy_convergence_threshold":energy_convergence_threshold,
                  "sweep_schedule_bond_dims":sweep_schedule_bond_dims,
                  "sweep_schedule_noise": sweep_schedule_noise,
                  "sweep_schedule_davidson_threshold":sweep_schedule_davidson_threshold,
                  "davidson_type": None,  # Default is None, for "Normal"
                  "eigenvalue_cutoff": 1e-120,
                  "davidson_max_iterations": 4000,  # Default is 4000
                  "davidson_max_krylov_subspace_size": 50,  # Default is 50
                  "lowmem_noise_bool": False,
                  "sweep_start": 0,  # Default is 0, where to start sweep
                  "initial_sweep_direction": None,
                  "stack_mem_ratio": 0.4,  # Default is 0.4
                  }
    return dmrg_param


default_final_bond_dim_type1 = 100
default_sweep_schedule_bond_dims_type1 = [default_final_bond_dim_type1] * 4 + [
    default_final_bond_dim_type1
] * 4
default_sweep_schedule_noise_type1 = [1e-4] * 4 + [1e-5] * 4 + [0] # [0] * 10
default_sweep_schedule_davidson_threshold_type1 = [1e-20] * 8

init_state_bond_dimension_1 = 100
max_num_sweeps_1 = 200
energy_convergence_threshold_1 = 1e-8
sweep_schedule_bond_dims_1 = default_sweep_schedule_bond_dims_type1
sweep_schedule_noise_1 = default_sweep_schedule_noise_type1
sweep_schedule_davidson_threshold_1 = (
    default_sweep_schedule_davidson_threshold_type1
)

def get_dmrg_config_type1(num_orbitals, num_electrons, num_unpaired_electrons,
                   multiplicity):
    nuc_rep_energy = 0
    dmrg_param = {"factor_half_convention": True,
                  "symmetry_type": "SZ",
                  "num_threads": 1,
                  "n_mkl_threads": 1,
                  "num_orbitals": num_orbitals,
                  "num_spin_orbitals": 2 * num_orbitals,
                  "num_electrons": num_electrons,
                  "two_S": num_unpaired_electrons,
                  "two_Sz": int((multiplicity - 1) / 2),
                  "orb_sym": None,
                  "temp_dir": "./temp",
                  "stack_mem": 1073741824,
                  "restart_dir": "./restart",
                  "core_energy": nuc_rep_energy,
                  "reordering_method": "none",
                  "init_state_seed": 64241,  # 0 means random seed
                  "initial_mps_method": "random",
                  "init_state_bond_dimension": init_state_bond_dimension_1,
                  "occupancy_hint": None,
                  "full_fci_space_bool": True,
                  "init_state_direct_two_site_construction_bool": False,
                  "max_num_sweeps": max_num_sweeps_1,
                  "energy_convergence_threshold":energy_convergence_threshold_1,
                  "sweep_schedule_bond_dims":sweep_schedule_bond_dims_1,
                  "sweep_schedule_noise": sweep_schedule_noise_1,
                  "sweep_schedule_davidson_threshold":sweep_schedule_davidson_threshold_1,
                  "davidson_type": None,  # Default is None, for "Normal"
                  "eigenvalue_cutoff": 1e-120,
                  "davidson_max_iterations": 4000,  # Default is 4000
                  "davidson_max_krylov_subspace_size": 50,  # Default is 50
                  "lowmem_noise_bool": False,
                  "sweep_start": 0,  # Default is 0, where to start sweep
                  "initial_sweep_direction": None,
                  "stack_mem_ratio": 0.4,  # Default is 0.4
                  }
    return dmrg_param


def get_dmrg_process_param():
    """
    Get parameters for dmrg process
    Returns:

    """
    dmrg_process_param = {
        "init_state_bond_dimension": init_state_bond_dimension,
        "max_num_sweeps": max_num_sweeps,
        "energy_convergence_threshold": energy_convergence_threshold,
        "sweep_schedule_bond_dims": sweep_schedule_bond_dims,
        "sweep_schedule_noise": sweep_schedule_noise,
        "sweep_schedule_davidson_threshold": sweep_schedule_davidson_threshold
    }
