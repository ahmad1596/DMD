#!/bin/bash

/usr/sbin/i2cset -y 2 0x1b 0x1F 0x00 0x00 0x00 0x01 i
/usr/sbin/i2cset -y 2 0x1b 0x21 0x00 0x00 0x00 0x01 i
/usr/sbin/i2cset -y 2 0x1b 0x1E 0x00 0x00 0x00 0x01 i
/usr/sbin/i2cset -y 2 0x1b 0xa3 0x00 0x00 0x00 0x01 i
/usr/sbin/i2cset -y 2 0x1b 0x7e 0x00 0x00 0x00 0x02 i
/usr/sbin/i2cset -y 2 0x1b 0x50 0x00 0x00 0x00 0x06 i
/usr/sbin/i2cset -y 2 0x1b 0x5e 0x00 0x00 0x00 0x00 i
/usr/sbin/i2cset -y 2 0x1b 0xB2 0x00 0x00 0x00 0x01 i
/usr/sbin/i2cset -y 2 0x1b 0xB3 0x00 0x00 0x00 0x01 i

