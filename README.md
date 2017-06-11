LUT Net
Proof of concept for making horribly non-differentiable things differentiable.

Ideas:

When creating the random connections, have a probability distribution for the connections.
This allows me to weight the MSBs higher than the LSBs for pixel values. Have the MSB 2 x more than (MSB-1) which is 2X more than (MSB-2) etc. See what XORnet does

Have a linebuffered pipeline architecture to do CNNs. This means sharing the LUTs for each layer of the Net in the same way that normal CNNs work. 
