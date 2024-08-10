### On the Discrete Cosserat Model for Soft Multisection Manipulators.

Soft manipulators, inspired by the functional role of  living organisms' soft tissues, provide better active environmental interaction  and compliance compared to rigid robot manipulators in society. Their designs, inspired by the nervous and musculoskeletal systems of living organic matter, enable their configurability and conformance. In particular, soft robot configurations, inspired by muscular hydrostats such as the Octopus robot, provide a nice interplay between continuum mechanics and sensorimotor control. These mechanisms are an excellent testbed for the validation of dynamic models and control laws/neural policies. 

This codebase contains various models and controllers for an Octopus robot arm based on the Cosserat Brothers' "Theory of Deformable Structures" as proposed in the following papers:

+ [Lagrangian Properties and Control of Soft Robots Modeled with Discrete Cosserat Rods.](https://scriptedonachip.com/downloads/Papers/SoRoPD.pdf) Molu, Lekan and Chen, Shaoru and Sedal, Audrey. Fall 2023.

+ [Composite Fast-Slow Backstepping Design for Nonlinear Singularly Perturbed Newton-Euler Dynamics: Application to Soft Robots.](https://scriptedonachip.com/downloads/Papers/SoRoSPT.pdf) Molu, Lekan. Fall 2023.

+ [Boundary Control of  Multisection Soft Manipulators: A Case in PDE Backstepping Design.](https://scriptedonachip.com/downloads/Papers/SoRoBC.pdf) Molu, Lekan and Others, Fall 2023.


The arms's configuration is shown in the left inset below and the linear controller proposed for strain dynamics regulation in the first paper above has the block diagram (right in the inset below).

<div align="center">
 <img src="/resources/pcs_kine.jpg" height="250px">
 <img src="/resources/pd_cont.jpg" height="250px">
</div>

Some results on strain states regulation as reported in the linear control paper above are quickly illustrated below.

Cable-driven, strain twist setpoint terrestrial control & Fluid-actuated, strain twist setpoint  terrestrial control. 

<div align="center">
  <img src="/resources/072423_cable_4pcs_10N.jpg" height="250px" width="350px">
  <img src="/resources/072723_with_drag_4_pieces_tipload.10N_PD.jpg" height="250px" width="350px">
</div>

 Fluid-actuated, strain twist setpoint underwater control. & Cable-driven, strain twist setpoint  regulation.

<div align="center">
  <img src="/resources/072723_with_drag_4_pieces_tipload.10N_PD.jpg" height="250px" width="350px">
  <img src="/resources/090623_with_cable_10_pieces_tipload.0.2N_PD.jpg" height="250px" width="350px">
</div>

Here are the results of position control with no gravity-compensation & Gravity-compensated terrestrial position control.

<div align="center">
  <img src="/resources/072423_pos_control.jpg" height="250px" width="350px">
  <img src="/resources/fixed_fluid_pos.jpg" height="250px" width="350px">
</div>

 ### Prerequisites

This code was tested on a Linux kernel 5.15.0-83 running Ubuntu 20.04 Distro. It is written to be CUDA-agnostic but having a CUDA-capable GPU would significantly hasten the solve time of the simulations. The author used a `Conda` environment with `python 3.8` running `CUDA SDK version 11.3.20210513` and a an `NVIDIA Driver Version: 525.125.06`. 

To install the needed library components, change to the python subdirectory and do: 

```bash
  user@usergroup:~/DCM/python$  pip install -r requirements.txt
```

+ If you will be using `CUDA`, we defer all GPU arithmetic operations to [PyTorch](https://pytorch.org/). Head over to the [install page](https://pytorch.org/get-started/locally/), choose the queries that correspond to your system hardware/configuration to get up and running (we recommend using `conda` for all installations). 


### Basic Usage

Our goal is to regulate the strain and strain velocity states of the robot per section under different constant vertical tip loads _despite the inevitable non-constant loads due to gravity, external forces, and inertial forces_ given the nature of the soft manipulator. Seeking to mitigate the time-varying oscillations identified in~\cite{RendaTRO18}, we test the efficacy of our controllers on a (constant) setpoint (or time-varying trajectory) per experiment whereupon we track unit linear and angular strains in the $+y$ direction for all tip loads. For trajectory tracking, we choose sinusoidal signals as a function of the simulation time for the strain and its twist fields acting along the $+y$ direction. 

+ To run, first clone this repo recursively 

```bash
  user@usergroup:$ git clone git@github.com:robotsorcerer/DCM.git --recursive
  user@usergroup:$ cd DCM
```

Then, run the `dc_main.py` file

```bash
  user@usergroup:~/DCM/python$ python dc_main.py -integrator <fehlberg2|RK45|DOPRI5> -controller <PD|PID|backstep> [-gain_prop <float>] [-gain_deriv <float>] 
                                [-gain_integ <float>] [-verbose] [-tip_load <float>] [-num_pieces <int>] [-with_cable <or -nowithcable>] 
                                [-with_drag <or -nowithdrag>] [-with_grav <or -nowithgrav>] [-atol <float>] [-rtol <float>]
                                [-desired_strain <float>] [-reference <setpoint|trajtrack>] 
```

Or you could just run multiple experiments using the [benchmarks.sh](python/benchmarks.sh) file. 
This dumps a bunch of simulation experiments to your `/opt/SoRo<-controller>` folder.

```bash
  user@usergroup:~/DCM/python$ ./benchmarks.sh -integrator <fehlberg2|RK45|DOPRI5> -controller <PD|PID|backstep> [-gain_prop <float>] [-gain_deriv <float>] 
                                [-gain_integ <float>] [-verbose] [-tip_load <float>] [-num_pieces <int>] [-with_cable <or -nowithcable>] 
                                [-with_drag <or -nowithdrag>] [-with_grav <or -nowithgrav>] [-atol <float>] [-rtol <float>]
                                [-desired_strain <float>] [-reference <setpoint|trajtrack>] 
```

Please look within the main file for the example options used for the gains and other parameters.


#### PID Controller 

Here's an example with 6 soft robot serially adjoined pieces under a PID controller that accounts for buoyancy-gravitational compensation with a cable-driven actuation underwater for proportional, integral, and derivative gains of 30, 1.2 and 3.5 respectively.

```bash
  user@usergroup:~/DCM/python$ python dc_main.py --num_pieces 6 --with_cable --with_drag --with_grav --controller PID --gain_prop 30 --gain_deriv 3.5 --gain_integ 1.2
```

To turn off one of the environments under which the controller is being simulated, simply ignore the flag that corresponds to the argument you do not want. 

#### PD Controller 

Same instruction as above save we pass in `PD` to the controller argument.

```bash
  user@usergroup:~/DCM/python$ python dc_main.py --num_pieces 6 --with_cable --with_drag --with_grav --controller PD --gain_prop 30 --gain_deriv 3.5 --gain_integ 1.2
```

#### Backstepping Controller 

This is where it gets interesting. This nonlinear controller has an attitude that screams clean steady state convergence throughout the state space. To deploy, go like so:

```bash
  user@usergroup:~/DCM/python$ python dc_main.py --num_pieces 6 --with_cable --with_drag --with_grav --controller backstep --gain_prop 30 --gain_deriv 3.5 --gain_integ 1.2
```

**Options**:
* `-gpu`:-- Default: `0`.
* `-num_pieces`:-- Piecewise constant pieces that comprise the deformable material. Default: `4`.
* `-controller`: Controller type: <`backstep` | `PD` | `PID`>. Default is `PID`.
* `-integrator`: Time integration scheme for the rhs of the Newton-Euler equation system. Options include <`felhberg2` | `dopri8` | `dopri5` | `euler` | `midpoint` | `rk4`>. Default is `fehlberg2`.
* `-reference`: What type of reference are we tracking? <`setpoint | trajtrack`>. Default is `setpoint`.
* `-with_cable`:-- Use cable actuation to generate motion? Default: `False`.
* `-with_drag`: Underwater simulation? Default: `False`.
* `-with_grav`:-- Compensate for gravity in the control law? Default: `False`.
* `-gain_prop`:-- Proportional gain for the PID/PD/Backstepping controller. Default: `3.5`.
* `-gain_deriv`:-- Derivative gain for the PID/PD controller. Default: `0.42`.
* `-gain_integ`:-- Integral gain for the PID/PD controller. Default: `0.1`.
* `-tip_load`:-- Constant vertical tip load that causes spatially varying forces throughout every section of the robot. Default: `10 Newtons` with $10^{-3}$ as a unit reference.


### Computational Speed 

Speed of computation can vary a lot depending on the GPU type and the integrator/optimizer. Here are some times for running various controllers (on an NVIDIA Quadro RTX 4000):

#### PD Controller

* `-integrator fehlberg2 -controller PD -gain_prop 50 -gain_deriv 0.1 -tip_load 10 -num_pieces 4 -with_drag False -with_cable: False`: `926.9554 mins or 15.4493 hours`;
* `-integrator fehlberg2 -controller PD -gain_prop 4.0 -gain_deriv 0.5 -tip_load 10 -num_pieces 4 -with_drag False -with_cable: False`: `675.6596 mins or  11.2610 hours`;
* `-integrator fehlberg2 -controller PD -gain_prop 3.5 -gain_deriv 0.34 -tip_load 10 -num_pieces 4 -with_drag False -with_cable: False`: `277.4110 mins or  4.6235 hours`;
* `-integrator fehlberg2 -controller PD -gain_prop 4.0 -gain_deriv 5.5 -tip_load 10 -num_pieces 4 -with_drag False -with_cable True`: `3513.3881 mins or  58.5565 hours`;
* `-integrator fehlberg2 -controller PD -gain_prop 4.0 -gain_deriv 0.5 -tip_load 10 -num_pieces 4 -with_drag True -with_cable False`: `661.5728 mins or  11.0262 hours`;
* `-integrator fehlberg2 -controller PD -gain_prop 6.5 -gain_deriv 0.52 -tip_load 10 -num_pieces 4 -with_drag True -with_cable True`: `417.6795 mins or  6.9613 hours`.

#### PID Controller

+ Coming in ...

#### Backstepping Controller

+ Coming in ...

#### Singularly Perturbed Backstepping Controller

+ Coming in ...


### Visualization

After finishing executing the control tasks, to visualize the configuration changes of the soft robot model under the control laws as it is brought to equilibrium, see our notes in [dcm_control.ipynb](notes/dcm_control.ipynb) or the file [pd_control_viz.py](visuals/pd_control_viz.py) to reproduce all results in the paper.


### To-Do's

* `Bring yuor own robot (BYOR)`:  A root locus/Nyquist simulator for checking the damping of the closed loop system under different gains is forthcoming to help users intelligently choose desirable gains for their robot parameters.

#### Singularly Pertubed Controllers

In order to decompose the system into multiple separate time scales that allows faster control e.g. for real-time operations, we go back to the singular perturbation approach of Tikhonov and Vasileva in the 1950's and do a rope-a-dope with Kokotovic's contributions in the '90's for nonlinear systems in hierarchically separating slow and fast dynamics within the system. We end up with a stabilizing set of controllers for the separated time scales which we then exploit in achieving real-time control of the deformable bodies.
