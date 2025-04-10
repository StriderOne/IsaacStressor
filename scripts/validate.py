import argparse
import subprocess
import gymnasium as gym

parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")

args_cli = parser.parse_args()

def main():
    # with open("results.csv", "w") as f:
    #     f.write("variation,success_rate\n")

    results = []
    for i in range(4):
        command = ["python", "-u","scripts/robomimic/play.py", 
                        "--task", str(args_cli.task), 
                        "--num_rollouts", str(args_cli.num_rollouts),
                        "--checkpoint", str(args_cli.checkpoint),
                        "--horizon", str(args_cli.horizon),
                        "--variation", str(i),
                        "--seed", str(args_cli.seed)
                        ]
        # result = subprocess.run(command,
        #                 capture_output=True, 
        #                 text=True)
        subprocess.run(command)
        # for line in result.stdout.splitlines():
        #     if "Success rate:" in line:
        #         print("Added!")
        #         # with open("results.csv", "a") as f:
        #         #     f.write(f"{i},{line.split(' ')[-1]}\n")
        #         break
            

if __name__ == "__main__":
    # run the main function
    main()
