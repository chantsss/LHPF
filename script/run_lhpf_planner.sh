cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=$1
BUILDER=$2
FILTER=$3
CKPT=$4
VIDEO_SAVE_DIR=$5

# CHALLENGE="closed_loop_nonreactive_agents"
# CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"
# for CHALLENGE in "open_loop_boxes" "closed_loop_nonreactive_agents" "closed_loop_reactive_agents"
# for CHALLENGE in "open_loop_boxes" "closed_loop_nonreactive_agents" "closed_loop_reactive_agents"
for CHALLENGE in "closed_loop_nonreactive_agents" "closed_loop_reactive_agents"
# for CHALLENGE in "closed_loop_reactive_agents"
do 
    python run_simulation.py \
        +simulation=$CHALLENGE \
        planner=$PLANNER \
        scenario_builder=$BUILDER \
        scenario_filter=$FILTER \
        worker=sequential \
        verbose=true \
        experiment_uid="lhpf_planner/$FILTER" \
        planner.lhpf_planner.render=false \
        planner.lhpf_planner.planner_ckpt="$CKPT_ROOT/$CKPT" \
        +planner.lhpf_planner.pure_leanrning=true \
        +planner.lhpf_planner.save_dir=$VIDEO_SAVE_DIR
done

