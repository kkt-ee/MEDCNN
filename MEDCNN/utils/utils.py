import time
import datetime

# from datetime import timedelta
def elapsedtime(start_time, end_time):
    """Compute elapsed time

    MEDCNN: Multiresolution Encoder-Decoder Convolutional Neural Network
    Copyright (C) 2025 Kishore Kumar Tarafdar
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    """
    # Calculate elapsed time
    elapsed_time_seconds = end_time - start_time

    # Convert seconds to a timedelta object
    delta = datetime.timedelta(seconds=elapsed_time_seconds)

    # Extract minutes, seconds, and milliseconds from the timedelta object
    minutes = delta.seconds // 60
    seconds = delta.seconds % 60
    milliseconds = delta.microseconds // 1000

    # Format the result
    formatted_time = "{:02}:{:02}:{:03}".format(minutes, seconds, milliseconds)
    print(f"Elapsed time: mm:ss:mss \t {formatted_time}")
    return formatted_time

## Somewhere above

# start_time=time.time()

## Time elapsed
# end_time = time.time()
# elapsedtime(start_time, end_time)



## time stamp
# import time
# import datetime
def timestamp(time):
    """Create a time stamp

    MEDCNN: Multiresolution Encoder-Decoder Convolutional Neural Network
    Copyright (C) 2025 Kishore Kumar Tarafdar
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    """
    # Get the current timestamp using time.time()
    timestamp = time

    # Convert the timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(timestamp)

    # Print the datetime object
    print(dt_object)
    # formatted_string = dt_object.strftime('%Y%m%d%H%M%S')
    formatted_string = dt_object.strftime('%Y%m%d')
    return formatted_string

# timestamp(time.time())