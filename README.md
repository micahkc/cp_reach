# cp_reach

**cp_reach** is a Python library for analyzing cyber-physical systems (CPS) by computing reachable sets using rigorous mathematical methods such as trigonometry, Linear Matrix Inequalities (LMIs), and Lyapunov theory. The library currently supports modeling and reachability analysis for basic quadrotor and rover systems.

## Features

- Compute reachable sets for CPS using proven mathematical techniques
- Supports basic quadrotor and rover dynamic models
- Utilizes LMIs and Lyapunov stability theory for formal guarantees
- Modular design for easy extension to other CPS platforms

## Installation

pip install cp_reach


## Usage

Here’s a simple example to get started with the quadrotor model:

    import cp_reach as cp
    from matplotlib import pyplot as plt

    config = {
        "eps_controller": 20,
        "emi_disturbance": 20,
        "heading": 0,
        "width": 0.3,
        "COM_height": 0.06,
        "turn_radius": 8
    }

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    cp.rover.simple_turn.emi_disturbance(config, ax1)


    fig2, ax2 = plt.subplots(figsize=(6, 6))
    cp.rover.simple_turn.roll_over(config, ax2)

## Supported Systems

- Quadrotor (basic model)

- Rover (basic model)

## Future Work

- Support for fixed-wing aircraft model

- Support for satellite model

- Enhanced computational efficiency and visualization tools

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE)
 file for details.
 ++++++++++++++++++++
## Contributing

Contributions are welcome! Please open issues or submit pull requests on GitHub.

## Contact
For questions or support, please contact [Micah Condie](mailto:mkcondie01@gmail.com)
