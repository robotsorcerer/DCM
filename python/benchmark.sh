#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

reference="setpoint"
integrator="fehlberg2"
num_pieces=10
controller="PD"
with_cable=True
with_drag=True
with_grav=True
# For rationale for these gains see the gains of 072423_10_30_23_with_cable_4_pieces_tipload.10N_PD_Control.npz
gain_prop=1.5 #4.0
gain_deriv=5.8 #5.5 
gain_integ=0.5
tip_load=10
desired_strain=0.5
rtol=1e-7
atol=1e-9
t_time=25

shopt -s nocasematch
set +e

while [ $# -gt 0 ]; do key="$1"
    case "$key" in
    -ct|--controller)
        CONTROLLER=$2
        shift # past argument
        shift # past value
        ;;
    -gpu|--gpu)
        GPU=$2
        shift
        shift
        ;;
    -np|--num_pieces)
        --num_pieces=$2
        shift
        shift
        ;;
    -vb|--verbose)
        verbose=$2
        shift
        shift
        ;;
    -it|--integrator)
        integrator=$2
        shift
        shift
        ;;
    -wc|--with_cable)
        with_cable=$2
        shift
        shift
        ;;
    -wd|--with_drag)
        with_drag=$2
        shift
        shift
        ;;
    -wg|--with_grav)
        with_grav=$2
        shift
        shift
        ;;
    -Fp|--tip_load)
        tip_load=$2
        shift
        shift
        ;;
    -rf|--reference)
        reference=$2
        shift
        shift
        ;;
    -Kp|--gain_prop)
        gain_prop=$2
        shift
        shift
        ;;
    -KD|--gain_deriv)
        gain_deriv=$2
        shift
        shift
        ;;
    -KI|--gain_integ)
        gain_integ=$2
        shift
        shift
        ;;
    *) # unknown option
        POSITIONAL+=("$1") # save in array for later use
        shift # past arg
        ;;
    esac
done
set -- "${POSITION[@]}" # restore positional parameters

# move everything to lower case
CONTROLLER=$(echo "$CONTROLLER" | tr '[:upper:]' '[:lower:]')
reference=$(echo "$reference" | tr '[:upper:]' '[:lower:]')

help(){
        echo -e "Usage: ./`basename $0` -controller <pd|pid|backstep> [-num_pieces <int>] [-with_cable <with_cable|nowith_cable>]
                \t [-with_drag <with_drag|nowith_drag>] [-with_grav <with_grav|nowith_grav>] [-gain_prop <float>]
                \t [-gain_deriv <float>] [-gain_integ <float>]."
        exit 1
}

print_opts(){
    echo -e "Default paramters: \n
            #===================================================================================================#
            \tController: $CONTROLLER | Reference: ${reference} | Pieces: ${num_pieces} |
            \t  Cable-driven: ${with_cable} | Underwater Ops: $with_drag | Gravity-compensation: ${with_grav} |
            \t  Gain_prop: ${gain_prop} | Gain_derivs: ${gain_deriv} | Integral gain: ${gain_integ}.
            #===================================================================================================#"
}

print_opts

run_controller() {
    python main.py "$@"
    wait
}

###### "PD Control Section" #######
# with grav and cable
pd_control(){
    #================================ Gravity compensation========================================#
    # grav | cable | drag ==> Cable driven underwater
    python main.py  -controller PD -reference $reference -integrator ${integrator} -num_pieces 10 \
                        -with_cable -with_drag -with_grav -gain_prop 2.5 -verbose \
                        -gain_deriv 5.3 -desired_strain ${desired_strain} -tip_load 1 \
                        -rtol ${rtol} -atol ${atol} -t_time ${t_time}
    
    # grav | cable | nodrag ==> Cable-driven terrestrial
    python main.py  --controller PD -reference $reference -integrator ${integrator} -num_pieces 5 \
                        -with_cable -nowith_drag -with_grav -gain_prop 1.5 -verbose \
                        -gain_deriv 5.8 -desired_strain 0.3 -tip_load 0.2 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # grav | nocable | nodrag ==> Fluid-driven terrestrial
    python main.py  -controller PD -reference trajtrack -integrator ${integrator} -num_pieces 7 \
                        -nowith_cable -nowith_drag -with_grav -gain_prop 5.5  -verbose  \
                        -gain_deriv 4.5 -desired_strain 0.45 -tip_load 2.5 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # grav | nocable | drag ==> Fluid-driven underwater
    python main.py  -controller PD -reference $reference -integrator ${integrator} -num_pieces 4\
                        -nowith_cable -with_drag -with_grav -gain_prop 6.0 \
                        -gain_deriv 5.5  -desired_strain 1.5 -tip_load 3.5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # grav | no cable | drag ==> Fluid-driven underwater
    python main.py  -controller PD -reference trajtrack -integrator ${integrator} -num_pieces 3 \
                        -nowith_cable -with_drag -with_grav -gain_prop 3.5 \
                        -gain_deriv 4.5 -desired_strain 3.5 -tip_load 5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    #======================= No Gravity Compensation =======================================#
    # nograv | no cable | nodrag   ===> Fluid-driven Terrestrial (No grav compensation)
    python main.py  -controller PD -reference trajtrack -integrator ${integrator} -num_pieces 7 \
                        -nowith_cable -nowith_drag -nowith_grav -gain_prop 5.5  -verbose  \
                        -gain_deriv 4.5 -desired_strain 0.45 -tip_load 2.5 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # no grav | cable | nodrag ===> Cable-driven Terrestrial (No grav compensation)
    python main.py  --controller PD -reference $reference -integrator ${integrator} -num_pieces 5 \
                        -with_cable -nowith_drag -nowith_grav -gain_prop 1.5 -verbose \
                        -gain_deriv 5.8 -desired_strain 0.3 -tip_load 0.2 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # no grav | no cable | drag ===> Fluid-driven Underwater (No grav compensation)
    python main.py  -controller PD -reference trajtrack -integrator ${integrator} -num_pieces 3 \
                        -nowith_cable -with_drag -nowith_grav -gain_prop 3.5 \
                        -gain_deriv 4.5 -desired_strain 3.5 -tip_load 5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # no grav | cable | drag ===> Cable-driven Underwater (No grav compensation)
    python main.py  -controller PD -reference $reference -integrator ${integrator} -num_pieces 10 \
                        -with_cable -with_drag -nowith_grav -gain_prop ${gain_prop} -verbose \
                        -gain_deriv ${gain_deriv} -desired_strain ${desired_strain} -tip_load 1 \
                        -rtol ${rtol} -atol ${atol} -t_time ${t_time}

    # no grav | no cable | nodrag   ===> Fluid Terrestrial (No grav compensation) ==> Cool experiment
    python main.py  -controller PD -reference $reference -integrator ${integrator} -num_pieces 4\
                        -nowith_cable -nowith_drag -nowith_grav -gain_prop 6.0 \
                        -gain_deriv 5.5  -desired_strain 1.5 -tip_load 3.5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

}

###### "PID Control Section" #######
pid_control(){
    #================================ Gravity compensation========================================#
    # grav | cable | drag ==> Cable driven underwater
    python main.py  -controller PID -reference $reference -integrator ${integrator} -num_pieces 10 \
                        -with_cable -with_drag -with_grav -gain_prop ${gain_prop} -verbose \
                        -gain_deriv ${gain_deriv} -desired_strain ${desired_strain} -tip_load 1 \
                        -rtol ${rtol} -atol ${atol} -t_time ${t_time}
    
    # grav | cable | nodrag ==> Cable-driven terrestrial
    python main.py  --controller PID -reference $reference -integrator ${integrator} -num_pieces 5 \
                        -with_cable -nowith_drag -with_grav -gain_prop 1.5 -verbose \
                        -gain_deriv 5.8 -desired_strain 0.3 -tip_load 0.2 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # grav | nocable | nodrag ==> Fluid-driven terrestrial
    python main.py  -controller PID -reference trajtrack -integrator ${integrator} -num_pieces 7 \
                        -nowith_cable -nowith_drag -with_grav -gain_prop 5.5  -verbose  \
                        -gain_deriv 4.5 -desired_strain 0.45 -tip_load 2.5 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # grav | nocable | drag ==> Fluid-driven underwater
    python main.py  -controller PID -reference $reference -integrator ${integrator} -num_pieces 4\
                        -nowith_cable -with_drag -with_grav -gain_prop 6.0 \
                        -gain_deriv 5.5  -desired_strain 1.5 -tip_load 3.5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # grav | no cable | drag ==> Fluid-driven underwater
    python main.py  -controller PID -reference trajtrack -integrator ${integrator} -num_pieces 3 \
                        -nowith_cable -with_drag -with_grav -gain_prop 3.5 \
                        -gain_deriv 4.5 -desired_strain 3.5 -tip_load 5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    #======================= No Gravity Compensation =======================================#
    # nograv | no cable | nodrag   ===> Fluid-driven Terrestrial (No grav compensation)
    python main.py  -controller PID -reference trajtrack -integrator ${integrator} -num_pieces 7 \
                        -nowith_cable -nowith_drag -nowith_grav -gain_prop 5.5  -verbose  \
                        -gain_deriv 4.5 -desired_strain 0.45 -tip_load 2.5 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # no grav | cable | nodrag ===> Cable-driven Terrestrial (No grav compensation)
    python main.py  --controller PID -reference $reference -integrator ${integrator} -num_pieces 5 \
                        -with_cable -nowith_drag -nowith_grav -gain_prop 1.5 -verbose \
                        -gain_deriv 5.8 -desired_strain 0.3 -tip_load 0.2 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # no grav | no cable | drag ===> Fluid-driven Underwater (No grav compensation)
    python main.py  -controller PID -reference trajtrack -integrator ${integrator} -num_pieces 3 \
                        -nowith_cable -with_drag -nowith_grav -gain_prop 3.5 \
                        -gain_deriv 4.5 -desired_strain 3.5 -tip_load 5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # no grav | cable | drag ===> Cable-driven Underwater (No grav compensation)
    python main.py  -controller PID -reference $reference -integrator ${integrator} -num_pieces 10 \
                        -with_cable -with_drag -nowith_grav -gain_prop ${gain_prop} -verbose \
                        -gain_deriv ${gain_deriv} -desired_strain ${desired_strain} -tip_load 1 \
                        -rtol ${rtol} -atol ${atol} -t_time ${t_time}

    # no grav | no cable | nodrag   ===> Fluid Terrestrial (No grav compensation) ==> Cool experiment
    python main.py  -controller PID -reference $reference -integrator ${integrator} -num_pieces 4\
                        -nowith_cable -nowith_drag -nowith_grav -gain_prop 6.0 \
                        -gain_deriv 5.5  -desired_strain 1.5 -tip_load 3.5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}
}

###### "Backstepping controller " ######
backstep_control(){
    #================================ Gravity compensation========================================#
    # grav | cable | drag ==> Cable driven underwater
    python main.py  -controller backstep -reference $reference -integrator ${integrator} -num_pieces 10 \
                        -with_cable -with_drag -with_grav -gain_prop ${gain_prop} -verbose \
                        -gain_deriv ${gain_deriv} -desired_strain ${desired_strain} -tip_load 1 \
                        -rtol ${rtol} -atol ${atol} -t_time ${t_time}
    
    # grav | cable | nodrag ==> Cable-driven terrestrial
    python main.py  --controller backstep -reference $reference -integrator ${integrator} -num_pieces 5 \
                        -with_cable -nowith_drag -with_grav -gain_prop 1.5 -verbose \
                        -gain_deriv 5.8 -desired_strain 0.3 -tip_load 0.2 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # grav | nocable | nodrag ==> Fluid-driven terrestrial
    python main.py  -controller backstep -reference trajtrack -integrator ${integrator} -num_pieces 7 \
                        -nowith_cable -nowith_drag -with_grav -gain_prop 5.5  -verbose  \
                        -gain_deriv 4.5 -desired_strain 0.45 -tip_load 2.5 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # grav | nocable | drag ==> Fluid-driven underwater
    python main.py  -controller backstep -reference $reference -integrator ${integrator} -num_pieces 4\
                        -nowith_cable -with_drag -with_grav -gain_prop 6.0 \
                        -gain_deriv 5.5  -desired_strain 1.5 -tip_load 3.5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # grav | no cable | drag ==> Fluid-driven underwater
    python main.py  -controller backstep -reference trajtrack -integrator ${integrator} -num_pieces 3 \
                        -nowith_cable -with_drag -with_grav -gain_prop 3.5 \
                        -gain_deriv 4.5 -desired_strain 3.5 -tip_load 5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    #======================= No Gravity Compensation =======================================#
    # nograv | no cable | nodrag   ===> Fluid-driven Terrestrial (No grav compensation)
    python main.py  -controller backstep -reference trajtrack -integrator ${integrator} -num_pieces 7 \
                        -nowith_cable -nowith_drag -nowith_grav -gain_prop 5.5  -verbose  \
                        -gain_deriv 4.5 -desired_strain 0.45 -tip_load 2.5 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # no grav | cable | nodrag ===> Cable-driven Terrestrial (No grav compensation)
    python main.py  --controller backstep -reference $reference -integrator ${integrator} -num_pieces 5 \
                        -with_cable -nowith_drag -nowith_grav -gain_prop 1.5 -verbose \
                        -gain_deriv 5.8 -desired_strain 0.3 -tip_load 0.2 \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # no grav | no cable | drag ===> Fluid-driven Underwater (No grav compensation)
    python main.py  -controller backstep -reference trajtrack -integrator ${integrator} -num_pieces 3 \
                        -nowith_cable -with_drag -nowith_grav -gain_prop 3.5 \
                        -gain_deriv 4.5 -desired_strain 3.5 -tip_load 5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}

    # no grav | cable | drag ===> Cable-driven Underwater (No grav compensation)
    python main.py  -controller backstep -reference $reference -integrator ${integrator} -num_pieces 10 \
                        -with_cable -with_drag -nowith_grav -gain_prop ${gain_prop} -verbose \
                        -gain_deriv ${gain_deriv} -desired_strain ${desired_strain} -tip_load 1 \
                        -rtol ${rtol} -atol ${atol} -t_time ${t_time}

    # no grav | no cable | nodrag   ===> Fluid Terrestrial (No grav compensation) ==> Cool experiment
    python main.py  -controller backstep -reference $reference -integrator ${integrator} -num_pieces 4\
                        -nowith_cable -nowith_drag -nowith_grav -gain_prop 6.0 \
                        -gain_deriv 5.5  -desired_strain 1.5 -tip_load 3.5  -verbose \
                        -rtol ${rtol} -atol ${atol}  -t_time ${t_time}
}

if [ "$CONTROLLER" == "pid" ]; then
    pid_control
elif [ "$CONTROLLER" == "pd" ]; then
    pd_control
elif [ "$CONTROLLER" = "backstep" ]; then
    backstep_control
else
    help
fi
