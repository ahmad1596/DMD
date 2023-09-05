# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import paramiko
import time
import os
import concurrent.futures

def setup_ssh_connection():
    hostname = "beaglebone"
    port = 22
    username = "debian"
    password = "temppwd"
    client = establish_ssh_connection(hostname, port, username, password)
    return client
    
def execute_command(client, command):
    stdin, stdout, stderr = client.exec_command(command)

def get_script_path(script_name):
    base_path = "/home/debian/boot-scripts-master/boot-scripts-master/device/bone/capes/DLPDLCR2000/"
    return os.path.join(base_path, script_name)

def initialise_switching(client):
    execute_command(client, f"/usr/bin/python2 {get_script_path('LEDSwitchOff.py')}")
    time.sleep(3)
    execute_command(client, get_script_path("deinitialise.sh"))
    time.sleep(3)
    execute_command(client, get_script_path("initialise.sh"))
    time.sleep(3)
    
def perform_dmd_switching_concurently(client):
    start_time_2 = time.time()
    execute_command(client,  f"/usr/bin/python2 {get_script_path('switching.py')}")
    end_time_2 = time.time()
    total_time = end_time_2 - start_time_2
    print(f"Total time taken by perform_dmd_switching_concurently: {total_time:.2f} seconds")


def establish_ssh_connection(hostname, port, username, password):
    client = paramiko.SSHClient()
    try:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, password)
        return client
    except paramiko.AuthenticationException as auth_ex:
        print("Authentication failed:", str(auth_ex))
    except paramiko.SSHException as ssh_ex:
        print("SSH connection failed:", str(ssh_ex))
    except Exception as ex:
        print("Error:", str(ex))
    return None

def run_tasks_concurrently(tasks):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_task = {executor.submit(task['function'], *task['args']): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                task['result'] = result
            except Exception as e:
                task['result'] = e
    return tasks

def print_task_results(completed_tasks):
    for task in completed_tasks:
        if 'result' in task:
            result = task['result']
            if isinstance(result, Exception):
                print(f"An error occurred in {task['function'].__name__}: {result}")
            else:
                print(f"{task['function'].__name__} completed successfully!")
        
def close_ssh_connection(client):
    if client:
        client.exec_command("exit")
        client.close()
        print("SSH connection closed.")
        
def cleanup_resources(client):
    if client:
        close_ssh_connection(client)
    
def main():
    try:
        client = setup_ssh_connection()
        print("SSH connection established successfully!")
        
        if client:
            tasks = [
                {'function': perform_dmd_switching_concurently, 'args': (client,)}
            ]

        completed_tasks = run_tasks_concurrently(tasks)
        print_task_results(completed_tasks)
        
    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    main()
