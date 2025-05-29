def run_fmu(fmu_file, sysbom_file):
    fmu = 'ExampleScenario.fmu'
    # Load parameters
    json_file = "params.json"
    with open(sysbom_file, 'r') as file:
        param_dict = json.load(file)


    # Monte-Carlo Simulation
    start_time = 0.0
    stop_time = 40.0
    tgrid = np.linspace(0, stop_time, 4001)

    for i in range(100):
        param_dict["rover.rnd_seed"] = i
        result = fmpy.simulate_fmu(fmu, start_time=start_time,
                            stop_time=stop_time, start_values = param_dict, output=["rover.rover_3d.phi", "rover.rover_3d.y", "rover.rover_3d.x"])
        x = np.array(result['rover.rover_3d.x'])
        y = np.array(result['rover.rover_3d.y'])
        plt.plot(-y,x, color='r', alpha = 0.1)


def plot_from_csv(csv_file):
    data = pd.read_csv(gazebo)
    x_meters = data["longitude"]
    y_meters = data["latitude"]
    plt.plot(x_meters,y_meters,zorder=10,color="orange")
    print(data.shape)

    data = pd.read_csv('example/gazebo_coordinates.csv')
    # Convert longitude and latitude to meters
    longitude = data["longitude"]
    x_ref = longitude[0]
    x_meters = []
    for x in longitude:
        x = (x-x_ref)*111000 - 5
        x_meters.append(x)

    latitude = data["latitude"]
    y_ref = latitude[0]
    y_meters = []
    for y in latitude:
        y = (y-y_ref)*-111000#*math.cos(x_ref*3.14/180)
        y_meters.append(y)

    print(x_meters)
    print(y_meters)
    
    plt.plot(y_meters,x_meters,zorder=10,color="orange")