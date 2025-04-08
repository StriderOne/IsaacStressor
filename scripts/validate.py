import subprocess
with open("results.csv", "w") as f:
    f.write("variation,success_rate\n")

results = []
for i in range(4):
    command = ["python", "-u","scripts/robomimic/play.py", 
                    "--task", "Isaac-Extended-Open-Drawer-Franka-IK-Rel-v0", 
                    "--num_rollouts", "50",
                    "--checkpoint", "/home/homa/IsaacStressor/logs/robomimic/Isaac-Extended-Open-Drawer-Franka-IK-Rel-v0/bc/20250405180542/models/model_epoch_2000.pth",
                    "--horizon", "1200",
                    "--variation", f"{i}",
                    ]
    if i != 1:
        command.append("--headless")
    result = subprocess.run(command,
                    capture_output=True, 
                    text=True)

    for line in result.stdout.splitlines():
        if "Success rate:" in line:
            print("Added!")
            with open("results.csv", "a") as f:
                f.write(f"{i},{line.split(' ')[-1]}\n")
            break
        

