# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import numpy as np
import matplotlib.pyplot as plt
import os

def create_output_folder(folder_name='mask_outputs'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def save_output_files(file_name, binary_array, micromirror_pitch, slits_type, option, slit_locations=None, alternate_size=None, array_size=None, slit_width=None, slit_spacing=None, unit_size=None, radius=None, center_coordinates=None, radius_inner=None, radius_outer=None):
    folder_name = create_output_folder()
    slits_type_mapping = {1: 'horizontal', 2: 'vertical', 3: 'circular', 4: 'cross', 5: 'ring'}  # Update slits_type_mapping
    file_name_without_suffix = f'{slits_type_mapping[slits_type]}'
    if slits_type == 3:
        file_name_without_suffix = 'circular'
        if radius is not None:
            file_name_without_suffix += f'_{radius}pixels_radius'
    elif slits_type == 5:  # Add this condition for the ring pattern
        if radius_inner is not None and radius_outer is not None:
            file_name_without_suffix = f'ring_radius_inner_{radius_inner}pixel_outer_{radius_outer}pixels'
    slit_positions_str = ''
    if option == 2:
        file_name_without_suffix = 'checkerboard'
    if option == 1: 
        if slit_locations is not None:
            slit_positions_str = '_'.join(map(str, slit_locations))
            file_name_without_suffix += f'_specific_positions_{slit_positions_str}'
        else:
            file_name_without_suffix += '_no_slits'
    elif option == 2:
        if alternate_size is not None:
            file_name_without_suffix = 'checkerboard'
        else:
            if slit_width is not None and slit_spacing is not None:
                if slits_type == 1:
                    file_name_without_suffix = f'horizontal_slit_width_{slit_width}pixels_spacing_{slit_spacing}pixels'
                elif slits_type == 2:
                    file_name_without_suffix = f'vertical_slit_width_{slit_width}pixels_spacing_{slit_spacing}pixels'
    if unit_size is not None:
        file_name_without_suffix += f'_{unit_size}pixels'
    if center_coordinates is not None:
        file_name_without_suffix += f'_({center_coordinates[0]},{center_coordinates[1]})'
    if slits_type == 5:
        file_name_display = f"{file_name_without_suffix}_radius_inner_{radius_inner}pixel_outer_{radius_outer}pixels_display"
    else:
        file_name_display = f"{file_name_without_suffix}_display"
    file_name_pixels = f"{file_name_without_suffix}"
    file_path_display = os.path.join(folder_name, f'{file_name_display}.png')
    file_path_pixels = os.path.join(folder_name, f'{file_name_pixels}.png')
    plot_binary_array_pixels(file_path_pixels, binary_array, array_size, slits_type, option, slit_locations, alternate_size)
    size_pixels = plt.imread(file_path_pixels).shape
    plot_binary_array_display(file_path_display, binary_array, micromirror_pitch, slits_type, option, slit_locations, alternate_size)
    print(f"\nFiles saved in folder '{folder_name}':")
    print(f"1. {file_name_display}.png (size: {array_size[1]} x {array_size[0]})")
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
    print(f"allowed {'horizontal' if slits_type == 1 else 'vertical'} slit locations: {start_location} to {end_location}")

def parse_slit_input(slit_input):
    if 'to' in slit_input:
        start, end = map(int, slit_input.split('to'))
        return list(range(start, end + 1))
    else:
        return [int(position) for position in slit_input.split(',')]

def generate_specific_positions(shape, slits_type, slit_positions):
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

def generate_alternate_slits(shape, slits_type, slit_width, slit_spacing, orientation='horizontal', checkerboard=False):
    binary_array = np.zeros(shape, dtype=np.uint8)
    if checkerboard:
        if slit_width == 1 and slit_spacing == 1:
            binary_array[::2, ::2] = 255
            binary_array[1::2, 1::2] = 255
        else:
            unit_width = slit_width + slit_spacing
            unit_height = slit_width + slit_spacing
            for i in range(0, shape[0], unit_height):
                for j in range(0, shape[1], unit_width):
                    binary_array[i:i + slit_width, j:j + slit_width] = 255  
    else:
        if slit_width == 1 and slit_spacing == 1:
            if orientation == 'horizontal':
                binary_array[::2, :] = 255
            elif orientation == 'vertical':
                binary_array[:, ::2] = 255
        else:
            if orientation == 'horizontal':
                for i in range(0, shape[0], slit_width + slit_spacing):
                    binary_array[i:i + slit_width, :] = 255
            elif orientation == 'vertical':
                for i in range(0, shape[1], slit_width + slit_spacing):
                    binary_array[:, i:i + slit_width] = 255
    return binary_array

def generate_circular_pattern(shape):
    binary_array = np.zeros(shape, dtype=np.uint8)
    center_x, center_y = shape[1] // 2, shape[0] // 2
    max_radius = min(center_x, center_y)
    print(f"\nAllowed radius values: 1 to {min(center_x, center_y)} pixels")
    while True:
        try:
            radius_input = input("Enter the radius of the circular pattern in pixels: ")
            radius = int(radius_input)
            if 1 <= radius <= max_radius:
                break
            else:
                print(f"Invalid input. Please enter a value between 1 and {max_radius}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")    
    y_coordinates, x_coordinates = np.ogrid[:shape[0], :shape[1]]
    mask = (x_coordinates - center_x) ** 2 + (y_coordinates - center_y) ** 2 <= radius ** 2
    pixels_used = np.sum(mask)
    binary_array[mask] = 255
    print(f"Circular pattern (radius: {radius} pixels) created using {pixels_used} pixels.")
    return binary_array, radius

def generate_cross_pattern(shape, intersection_point=None):
    binary_array = np.zeros(shape, dtype=np.uint8)
    if intersection_point is None:
        center_y = shape[0] // 2
        center_x = shape[1] // 2
    else:
        center_y, center_x = intersection_point
    binary_array[center_y, :] = 255
    binary_array[:, center_x] = 255
    print(f"Cross pattern created with intersection point at ({center_x},{center_y}).")
    return binary_array

def generate_ring_pattern(shape, radius_inner, radius_outer):
    binary_array = np.zeros(shape, dtype=np.uint8)
    center_x, center_y = shape[1] // 2, shape[0] // 2
    y_coordinates, x_coordinates = np.ogrid[:shape[0], :shape[1]]
    mask_inner = (x_coordinates - center_x) ** 2 + (y_coordinates - center_y) ** 2 <= radius_inner ** 2
    mask_outer = (x_coordinates - center_x) ** 2 + (y_coordinates - center_y) ** 2 <= radius_outer ** 2
    mask_ring = np.logical_and(mask_outer, np.logical_not(mask_inner))
    binary_array[mask_ring] = 255
    return binary_array

def plot_binary_array_display(file_path_display, binary_array, micromirror_pitch, slits_type, option, slit_coordinates=None, alternate_size=None):
    file_name = f'{slits_type}_'
    if option == 1:
        if slit_coordinates is not None:
            slit_positions_str = '_'.join(map(str, slit_coordinates))
            file_name += f'specific_positions_{slit_positions_str}'
        else:
            file_name += 'no_slits'
    elif option == 2:
        if alternate_size is not None:
            file_name += f'alternate_size_{alternate_size}'
        else:
            file_name += 'no_alternate_slits'
    fig, ax = plt.subplots(figsize=(640/600, 360/600), dpi=600)
    extent = [0, binary_array.shape[1] * micromirror_pitch, 0, binary_array.shape[0] * micromirror_pitch]
    ax.imshow(binary_array, cmap='gray', aspect='auto', extent=extent, interpolation='none', origin='lower')
    ax.axis('off')
    ax.set_title('')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(file_path_display, bbox_inches='tight', pad_inches=0)
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
    array_width, array_height = 640, 360
    print("\n*DLP2000*")
    print(f"Display Resolution: {array_width} x {array_height}")
    print(f"Display Dimension: {array_width * micromirror_pitch} um x {array_height * micromirror_pitch} um")
    while True:
        try:
            configuration_type = int(input("\nMicromirror Array Configuration:\n1. Horizontal slits\n2. Vertical slits\n3. Circular pattern\n4. Cross pattern\n5. Ring pattern\nChoose configuration (1, 2, 3, 4, or 5): "))
            if configuration_type not in [1, 2, 3, 4, 5]:
                raise ValueError("Invalid input. Please enter 1 for 'Horizontal slits', 2 for 'Vertical slits', 3 for 'Circular pattern', 4 for 'Cross pattern', or 5 for 'Ring pattern'.")
            break
        except ValueError as e:
            print(e)
    slits_type = None
    slit_locations = None
    alternate_size = None
    slit_width = None
    slit_spacing = None
    unit_size = None
    radius = None
    if configuration_type == 3:
        binary_array, radius = generate_circular_pattern((array_height, array_width))
        file_name = "circular_pattern"
        save_output_files(file_name, binary_array, micromirror_pitch, configuration_type, 0, slit_locations, alternate_size, (array_height, array_width), radius=radius)
        return
    if configuration_type == 4:
        center_x = array_width // 2
        center_y = array_height // 2
        print(f"\nDefault center for the cross pattern: ({center_x},{center_y})")
        intersection_point_input = input("Enter the intersection point for the cross pattern (e.g., x,y): ")
        if intersection_point_input:
            try:
                intersection_point = tuple(map(int, intersection_point_input.split(',')))
            except ValueError:
                print("Invalid input format. Please enter the intersection point as x,y.")
                return
        else:
            intersection_point = None
        binary_array = generate_cross_pattern((array_height, array_width), intersection_point)
        file_name = "cross_pattern"
        save_output_files(file_name, binary_array, micromirror_pitch, configuration_type, 0, None, None, (array_height, array_width), radius=radius, center_coordinates=(center_x, center_y))
        return
    if configuration_type == 5: 
      print("Maximum ring diameter: 360 pixels (display height).")
      while True:
          try:
              radius_inner_input = input("Enter the inner radius of the ring pattern in pixels: ")
              radius_outer_input = input("Enter the outer radius of the ring pattern in pixels: ")
              radius_inner = int(radius_inner_input)
              radius_outer = int(radius_outer_input)
              if 1 <= radius_inner < radius_outer:
                  break
              else:
                  print("Invalid input. Please ensure that 1 <= inner radius < outer radius.")
          except ValueError:
              print("Invalid input. Please enter valid integers.")
      binary_array = generate_ring_pattern((array_height, array_width), radius_inner, radius_outer)
      file_name = f"ring_pattern_radius_{radius_inner}_{radius_outer}"
      save_output_files(file_name, binary_array, micromirror_pitch, configuration_type, 0, None, None, (array_height, array_width),
                          radius_inner=radius_inner, radius_outer=radius_outer)
      return
    else:
        slits_type = configuration_type

    while True:
        print("\nChoose an option:")
        print("1. Specific positions")
        print("2. Alternate slits")
        option = input("Enter your choice (1 or 2): ")
        if option.isdigit() and int(option) in [1, 2]:
            option = int(option)
            break
        else:
            print("Invalid option. Please choose 1 for 'Specific positions' or 2 for 'Alternate slits'.")
    if option == 1:
        while True:
            print("\nChoose an option:")
            print("1. Single position")
            print("2. Multiple positions")
            print("3. Range of positions")
            slit_option = input("Enter your choice (1, 2, or 3): ")
            if slit_option.isdigit() and int(slit_option) in [1, 2, 3]:
                slit_option = int(slit_option)
                break
            else:
                print("Invalid option. Please choose 1 for 'Single position', 2 for 'Multiple positions', or 3 for 'Range of positions'.")
        slit_input_info = "Info: single position / position_1, position_2, position_3,... / range: 'start' to 'end'"
        print(slit_input_info)
        while True:
            try:
                slit_input = input(f"Enter slit location (allowed values: 0 to {array_height - 1}) for {'horizontal' if slits_type == 1 else 'vertical'} slits: ")
                slit_locations = parse_slit_input(slit_input)
                if (slits_type == 1 and any(slit < 0 or slit >= array_height for slit in slit_locations)) or (slits_type == 2 and any(slit < 0 or slit >= array_width for slit in slit_locations)):
                    raise ValueError(f"Invalid slit location. Please enter values in the range 0 to {array_height - 1}." if slits_type == 1 else f"Invalid slit location. Please enter values in the range 0 to {array_width - 1}.")
                else:
                    break
            except ValueError as e:
                print(e)
        binary_array = generate_specific_positions((array_height, array_width), slits_type, slit_locations)
    elif option == 2:
        while True:
            print("\nChoose an option:")
            print("1. Slits")
            print("2. Checkerboard")
            alternate_option = input("Enter your choice (1 or 2): ")
            if alternate_option.isdigit() and int(alternate_option) in [1, 2]:
                alternate_option = int(alternate_option)
                break
            else:
                print("Invalid option. Please choose 1 for 'Slits' or 2 for 'Checkerboard'.")
        if alternate_option == 1:
            orientation = 'vertical' if slits_type == 2 else 'horizontal'
            while True:
                try:
                    allowed_range = (1, array_height) if slits_type == 1 else (1, array_width)
                    slit_width_input = input(f"Enter pixel slit width (allowed values: {allowed_range[0]} to {allowed_range[1]}): ")
                    if not slit_width_input:
                        raise ValueError("Slit width cannot be empty. Please enter a value.")
                    slit_width = int(slit_width_input)
                    if not (allowed_range[0] <= slit_width <= allowed_range[1]):
                        raise ValueError(f"Invalid slit width. Please enter a value in the range {allowed_range[0]} to {allowed_range[1]}.")
                    break
                except ValueError as e:
                    print(e)
            while True:
                try:
                    slit_spacing_input = input(f"Enter pixel slit spacing (allowed values: 1 to {allowed_range[1]}): ")
                    if not slit_spacing_input:
                        raise ValueError("Slit spacing cannot be empty. Please enter a value.")
                    slit_spacing = int(slit_spacing_input)
                    if not (1 <= slit_spacing <= allowed_range[1]):
                        raise ValueError(f"Invalid slit spacing. Please enter a value in the range 1 to {allowed_range[1]}.")
                    break
                except ValueError as e:
                    print(e)
            binary_array = generate_alternate_slits((array_height, array_width), slits_type, slit_width, slit_spacing, orientation)
            file_name = f"{slits_type}_slits_width_{slit_width}pixel_spacing_{slit_spacing}pixel"
        elif alternate_option == 2:
            while True:
                try:
                    unit_size_input = input("Enter unit size (pixels per unit) in the checkerboard: ")
                    if not unit_size_input:
                        raise ValueError("Unit size cannot be empty. Please enter a value.")
                    unit_size = int(unit_size_input)
                    if unit_size <= 0:
                        raise ValueError("Invalid unit size. Please enter a positive value.")
                    break
                except ValueError as e:
                    print(e)
            binary_array = generate_alternate_slits((array_height, array_width), slits_type, unit_size, unit_size, checkerboard=True)
            file_name = f"checkerboard_unit_{unit_size}"
        else:
            print("Invalid option. Choose 1 for 'Slits' or 2 for 'Checkerboard'.")
            return
    else:
        print("Invalid option. Choose 1 for 'Specific positions' or 2 for 'Alternate slits'.")
        return
    if binary_array is not None:
        if option == 1:
            if slit_locations is not None:
                slit_positions_str = '_'.join(map(str, slit_locations))
                file_name = f"{slits_type}_specific_positions_{slit_positions_str}"
            else:
                file_name = f"{slits_type}_no_slits"
        elif option == 2:
            if alternate_option == 2:
                file_name = "checkerboard"
            else:
                file_name = f"{slits_type}_slits_width_{slit_width}pixel_spacing_{slit_spacing}pixel"
        else:
            print("Invalid option. Choose 1 for 'Specific positions' or 2 for 'Alternate slits'.")
            return
        save_output_files(file_name, binary_array, micromirror_pitch, slits_type, option, slit_locations, alternate_size, (array_height, array_width), slit_width, slit_spacing, unit_size)
if __name__ == "__main__":
    main()
