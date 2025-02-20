import pandas as pd
import pickle
import matplotlib.pyplot as plt
import glob
import csv
# Initialize a DataFrame to store graph data
data = []



# Convert to DataFrame and save
def save_results_to_pickle(filename):
    """
    param:
    filename: The name of the file to save the results to (str)
    """
    df = pd.DataFrame(data)
    df.to_pickle(filename)  # Save to a .pkl file for persistent storage

#store the results of the IP and Greedy variants in data frame
def store_results_IP_Greedy_variants(instance_size,commodities,IP_value,MGG_updated_value,MGG_remove_commodity_value,MGG_IP_combined_value):
    """
    param:
    instance_size: number of nodes (int)
    commodities: list of commodities - should be triples (P_i,w_i,c_i)i \in [1,k]
    IP_value: the value of the IP (int)
    MGG_updated_value: the value of the MGG_updated (int)
    MGG_remove_commodity_value: the value of the MGG_remove_commodity (int)
    MGG_IP_combined_value: the value of the MGG_IP_combined (int)
    """

    data.append({
        "instance_size":instance_size,
        "commodities": commodities,
        "IP_MGG_updated_value": IP_value / MGG_updated_value,
        "IP_MGG_remove_commodity_value": IP_value / MGG_remove_commodity_value,
        "IP_MGG_IP_combined_value": IP_value / MGG_IP_combined_value
    })


#load the results from the pickle files and plot them
def load_results_and_plot(pickle_directory_path):
    """
    param:
    pickle_directory_path: The path to the directory containing the pickle files (str)
    """

    # Define the path to the pickle files
    pickle_files_path = pickle_directory_path + '*.pkl'


    pickle_files = glob.glob(pickle_files_path)

    # Initialize lists to hold combined data
    x_combined = []
    y1_combined = []
    y2_combined = []
    y3_combined = []

    # Loop through each pickle file
    for pickle_file in pickle_files:
        # Load the data from the pickle file
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        x_combined.extend(data["instance_size"])
        y1_combined.extend(data["IP_MGG_updated_value"])
        y2_combined.extend(data["IP_MGG_IP_combined_value"])
        y3_combined.extend(data["IP_MGG_remove_commodity_value"])

    # Sort the data by instance size
    sorted_data = sorted(zip(x_combined, y1_combined, y2_combined, y3_combined))
    x_combined, y1_combined, y2_combined, y3_combined = zip(*sorted_data)


    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot each combined column
    plt.plot(x_combined, y1_combined, label=r"MGG$_{\mathrm{u}}$")
    plt.plot(x_combined, y2_combined,label=r"MGG$_{\mathrm{IP}}$")
    plt.plot(x_combined, y3_combined,label=r"MGG$_{\mathrm{RC}}$")


    # Add labels, title, and legend
    plt.legend()
    plt.grid()
    plt.xlabel('Number of nodes',fontsize=12)
    plt.ylabel(r'$\frac{c_{IP}(I)}{c_{MGG}(I)}$',fontsize=18,rotation=0,labelpad=16)
    # Save the figure
    #plt.savefig("new_runtime_analysis2/Big_test_set_results2.png", format='png',dpi=300)

    plt.show()
    

#read the runtimes from the csv file and plot them
def read_runtimes_and_plot(filename):
    """
    Plots three graphs: one with IP-values and all MGG variants, one with IP and MGG_RC, and one with MGG_u and MGG_IP
    param:
    filename: The name of the file to read the runtimes from (str)
    """

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header row
        data = [row for row in reader]

    # Convert data to appropriate types if necessary
    instance_sizes = [int(row[0]) for row in data]
    ip_time_total = [float(row[1]) for row in data]
    mgg_1_time_total = [float(row[2]) for row in data]
    Greedy_IP_combined_time_total = [float(row[3]) for row in data]
    mgg_3_time_total = [float(row[4]) for row in data]
    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot each column
    plt.plot(instance_sizes, ip_time_total, label="IP",color="blue")
    plt.plot(instance_sizes, mgg_1_time_total, label=r"MGG$_{\mathrm{u}}$",color="green")
    plt.plot(instance_sizes, Greedy_IP_combined_time_total, label=r"MGG$_{\mathrm{IP}}$",color="red")
    plt.plot(instance_sizes, mgg_3_time_total, label=r"MGG$_{\mathrm{RC}}$",color="orange")


    # Add labels, title, and legend
    plt.xlabel("Number of nodes")
    plt.ylabel("Runtime (seconds)")

    plt.legend()

    # Show the grid for better readability
    plt.grid(True)
    #plt.savefig("total_runtime_plot.png", dpi=300)
    # Display the plot
    plt.show()
    


    # Create the first plot with IP and MGG_RC
    plt.figure(figsize=(10, 6))
    plt.plot(instance_sizes, ip_time_total, label="IP",color="blue")
    plt.plot(instance_sizes, mgg_3_time_total, label=r"MGG$_{\mathrm{RC}}$",color="orange")

    plt.xlabel('Instance Size')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.grid(True)
    #plt.savefig('IP_MGG_RC_plot.png', dpi=300)
    plt.show()

    # Create the second plot with MGG_u and MGG_IP
    plt.figure(figsize=(10, 6))
    plt.plot(instance_sizes, mgg_1_time_total, label=r"MGG$_{\mathrm{u}}$",color="green")
    plt.plot(instance_sizes, Greedy_IP_combined_time_total, label=r"MGG$_{\mathrm{IP}}$",color="red")

    plt.xlabel('Instance Size')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Test2_MGG_u_MGG_IP_plot.png', dpi=300)
    plt.show()
