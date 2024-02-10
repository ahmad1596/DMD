import numpy as np
import matplotlib.pyplot as plt
import os

def create_output_folder(folder_name='mask_outputs'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def save_output_files(file_name, binary_array, micromirror_pitch, slits_type, option, slit_coordinates=None, alternate_size=None, array_size=None):
    folder_name = create_output_folder()
    slits_type_mapping = {1: 'horizontal', 2: 'vertical'}
    file_name_without_suffix = f'{slits_type_mapping[slits_type]}_'
    if option == 1:
        if slit_coordinates is not None:
            slit_positions_str = '_'.join(map(str, [slit_coordinates]))
            file_name_without_suffix += f'individual_positions_{slit_positions_str}'
        else:
            file_name_without_suffix += 'no_slits'
    elif option == 2:
        if alternate_size is not None:
            file_name_without_suffix += f'alternate_slits_spacing_{alternate_size}pixels'
        else:
            file_name_without_suffix += 'no_alternate_slits'
    if option == 1:
        file_name_um = file_name_without_suffix + '_um'
        file_name_pixels = file_name_without_suffix + '_pixels'
    elif option == 2:
        file_name_um = file_name_without_suffix + '_um'
        file_name_pixels = file_name_without_suffix + '_pixels'
    else:
        print("Invalid option. Choose 1 for 'Individual position' or 2 for 'Alternate slits'.")
        return
    file_path_um = os.path.join(folder_name, f'{file_name_um}.png')
    file_path_pixels = os.path.join(folder_name, f'{file_name_pixels}.png')
    plot_binary_array_pixels(file_path_pixels, binary_array, array_size, slits_type, option, slit_coordinates, alternate_size)
    size_pixels = plt.imread(file_path_pixels).shape
    plot_binary_array_um(file_path_um, binary_array, micromirror_pitch, slits_type, option, slit_coordinates, alternate_size)
    print(f"Files saved in folder '{folder_name}':")
    print(f"1. {file_name_um}.png (size: {array_size[1]} x {array_size[0]})")
    print(f"2. {file_name_pixels}.png (size: {size_pixels[1]} x {size_pixels[0]})")

def generate_slits(shape, slits_type, slit_coordinates, alternate_size=None):
    binary_array = np.zeros(shape, dtype=np.uint8)
    if slits_type == 1:
        binary_array[:, ::alternate_size] = 255 if alternate_size is not None else 0
    elif slits_type == 2:
        binary_array[::alternate_size, :] = 255 if alternate_size is not None else 0
    else:
        print("Invalid slits type. Choose 1 for 'vertical' or 2 for 'horizontal'.")
        return None
    return binary_array

def print_allowed_locations(slits_type, array_size):
    start_location, end_location = (0, array_size[0] - 1) if slits_type == 1 else (0, array_size[1] - 1)
    print(f"Allowed {'horizontal' if slits_type == 1 else 'vertical'} slit locations: {start_location} to {end_location}")

def parse_slit_input(slit_input):
    if 'to' in slit_input:
        start, end = map(int, slit_input.split('to'))
        return list(range(start, end + 1))
    else:
        return [int(position) for position in slit_input.split(',')]

def generate_individual_positions(shape, slits_type, slit_positions):
    binary_array = np.zeros(shape, dtype=np.uint8)
    if slits_type == 1:
        y_coordinates, x_coordinates = np.meshgrid(slit_positions, np.arange(shape[1]), indexing='ij')
    elif slits_type == 2:
        y_coordinates, x_coordinates = np.meshgrid(np.arange(shape[0]), slit_positions, indexing='ij')
    else:
        print("Invalid slits type. Choose 1 for 'vertical' or 2 for 'horizontal'.")
        return None
    binary_array[y_coordinates, x_coordinates] = 255
    return binary_array

def generate_alternate_slits(shape, slits_type, alternate_size, orientation='horizontal'):
    binary_array = np.zeros(shape, dtype=np.uint8)
    if orientation == 'horizontal':
        if slits_type == 1:
            y_coordinates, x_coordinates = np.meshgrid(np.arange(0, shape[0], alternate_size), np.arange(shape[1]), indexing='ij')
        elif slits_type == 2:
            y_coordinates, x_coordinates = np.meshgrid(np.arange(shape[0]), np.arange(0, shape[1], alternate_size), indexing='ij')
        else:
            print("Invalid slits type. Choose 1 for 'vertical' or 2 for 'horizontal'.")
            return None
    elif orientation == 'vertical':
        if slits_type == 1:
            y_coordinates, x_coordinates = np.meshgrid(np.arange(0, shape[0], alternate_size), np.arange(shape[1]), indexing='ij')
        elif slits_type == 2:
            y_coordinates, x_coordinates = np.meshgrid(np.arange(shape[0]), np.arange(0, shape[1], alternate_size), indexing='ij')
        else:
            print("Invalid slits type. Choose 1 for 'vertical' or 2 for 'horizontal'.")
            return None
    else:
        print("Invalid orientation. Choose 'horizontal' or 'vertical'.")
        return None
    binary_array[y_coordinates, x_coordinates] = 255
    return binary_array

def plot_binary_array_um(file_path_um, binary_array, micromirror_pitch, slits_type, option, slit_coordinates=None, alternate_size=None):
    file_name = f'{slits_type}_'
    if option == 1:
        if slit_coordinates is not None:
            slit_positions_str = '_'.join(map(str, slit_coordinates))
            file_name += f'individual_positions_{slit_positions_str}'
        else:
            file_name += 'no_slits'
    elif option == 2:
        if alternate_size is not None:
            file_name += f'alternate_size_{alternate_size}'
        else:
            file_name += 'no_alternate_slits'
    fig, ax = plt.subplots( figsize = (640/600, 360/600), dpi=600)
    extent = [0, binary_array.shape[1] * micromirror_pitch, 0, binary_array.shape[0] * micromirror_pitch]
    ax.imshow(binary_array, cmap='gray', aspect='auto', extent=extent, interpolation='none', origin='lower')
    ax.axis('off')
    ax.set_title('')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(file_path_um, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_binary_array_pixels(file_path_pixels, binary_array, array_size, slits_type, option, slit_coordinates=None, alternate_size=None):
    fig, ax = plt.subplots(figsize=(640/100, 360/100), dpi=600)
    ax.imshow(binary_array, cmap='gray', aspect='auto', extent=[0, array_size[1], 0, array_size[0]], interpolation='none', origin='lower')
    ax.set_xlabel('X (pixels)', fontsize=14, fontweight="bold")
    ax.set_ylabel('Y (pixels)', fontsize=14, fontweight="bold")
    x_ticks = np.arange(0, array_size[1] + 1, 50)
    y_ticks = np.arange(0, array_size[0] + 1, 50)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.set_title('', fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(file_path_pixels)
    plt.show()

def main():
    micromirror_pitch = 7.56
    array_height, array_width = 360, 640

    print("*DLP2000*")
    print(f"Display Resolution: {array_width} x {array_height}")
    print(f"Display Dimension: {array_width * micromirror_pitch} um x {array_height * micromirror_pitch} um")
    print(f"Micromirror Pitch: {micromirror_pitch} um")

    while True:
        try:
            slits_type = int(input("\nMicromirror Array Configuration:\n1. Horizontal slits\n2. Vertical slits\nEnter the type of slits (1 or 2): "))
            if slits_type not in [1, 2]:
                raise ValueError("Invalid input. Please enter 1 for 'Horizontal' or 2 for 'Vertical'.")
            break
        except ValueError as e:
            print(e)

    allowed_locations = f"Allowed {'horizontal' if slits_type == 1 else 'vertical'} slit locations: 0 to {array_width - 1}" if slits_type in [1, 2] else "Invalid slits type. Choose 1 for 'Horizontal' or 2 for 'Vertical'."
    print(allowed_locations)

    while True:
        option = input("Choose an option:\n1. Individual position\n2. Alternate slits\nEnter your choice (1 or 2): ")

        if option.isdigit() and int(option) in [1, 2]:
            option = int(option)
            break
        else:
            print("Invalid option. Please choose 1 for 'Individual position' or 2 for 'Alternate slits'.")

    alternate_size = None
    slit_coordinates = None

    if option == 1:
        slit_input_info = "Info: single position / position_1, position_2, position_3,... / range 'start to end'"
        print(slit_input_info)
        while True:
            slit_input = input(f"Enter slit location (Allowed values: 0 to {array_width - 1}) for {'horizontal' if slits_type == 1 else 'vertical'} slits: ")
            slit_locations = parse_slit_input(slit_input)

            if (slits_type == 1 and any(slit < 0 or slit >= array_width for slit in slit_locations)) or (slits_type == 2 and any(slit < 0 or slit >= array_height for slit in slit_locations)):
                print(f"Invalid slit location. Please enter values in the range 0 to {array_width - 1}." if slits_type == 1 else f"Invalid slit location. Please enter values in the range 0 to {array_height - 1}.")
            else:
                slit_coordinates = slit_locations
                break

        binary_array = generate_individual_positions((array_height, array_width), slits_type, slit_locations)
    elif option == 2:
        while True:
            allowed_range = (1, array_height) if slits_type == 1 else (1, array_width)
            alternate_size = int(input(f"Enter pixel slit spacing (allowed values: {allowed_range[0]} to {allowed_range[1]}): "))
            if not (allowed_range[0] <= alternate_size <= allowed_range[1]):
                print(f"Invalid alternate slit size. Please enter a value in the range {allowed_range[0]} to {allowed_range[1]}.")
            else:
                break

        binary_array = generate_alternate_slits((array_height, array_width), slits_type, alternate_size)

    else:
        print("Invalid option. Choose 1 for 'Individual position' or 2 for 'Alternate slits'.")
        return

    if binary_array is not None:
        if option == 1:
            slit_positions_str = '_'.join(map(str, slit_coordinates)) if slit_coordinates is not None else ''
            slit_positions_str = slit_positions_str.replace(',', '_')
            file_name = f"{slits_type}_individual_positions_{slit_positions_str}"
        elif option == 2:
            file_name = f"{slits_type}_alternate_size_{alternate_size}"
        else:
            print("Invalid option. Choose 1 for 'Individual position' or 2 for 'Alternate slits'.")
            return

        save_output_files(file_name, binary_array, micromirror_pitch, slits_type, option, slit_coordinates, alternate_size, (array_height, array_width))

if __name__ == "__main__":
    main()
