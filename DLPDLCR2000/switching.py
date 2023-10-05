import smbus
import threading
import time

# Function to set the mask via I2C
def set_mask(mask):
    bus = smbus.SMBus(2)
    address = 0x1b
    register = 0x2D
    data = [0x00, 0x00, 0x00, mask]
    bus.write_i2c_block_data(address, register, data)

# Function to switch the mask in a loop
def mask_switching_loop():
    number_of_masks = 10
    display_period = 0.018  # adjust as needed # actual minus 0.002
# for example, 0.020, display_period = 0.020 - 0.002 = 0.018
    for counter in range(number_of_masks + 1):
        mask = 0x00 if counter % 2 == 0 else 0x01
        print "{0} is {1}".format(counter, 'even' if mask == 0x00 else 'odd')
        set_mask(mask)
        
        # Wait for the display period
        time.sleep(display_period)

if __name__ == "__main__":
    # Start the mask switching loop in a background thread
    switch_thread = threading.Thread(target=mask_switching_loop)
    switch_thread.start()

    # Wait for the background thread to complete
    switch_thread.join()

