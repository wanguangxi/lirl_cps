"""
Compatibility shim for historical imports.

Some scripts (e.g. `RMS/algs/hyar_policy.py`) import symbols from a module named
`ddpg_lirl_pi`. In this repo, the implementation lives in `lirl.py`.

This file re-exports the relevant objects so users don't need to copy/rename files.
"""

from lirl import (  # noqa: F401
    CONFIG,
    MuNet,
    OrnsteinUhlenbeckNoise,
    QNet,
    ReplayBuffer,
    action_choose,
    action_projection,
    construct_constraints_for_qp,
    evaluate_multi_run_results,
    main,
    multi_run_training,
    plot_multi_run_training_curves,
    plot_training_curve,
    save_multi_run_results,
    save_results,
    solve_quadratic_program,
    soft_update,
    test_and_visualize,
    train,
)


