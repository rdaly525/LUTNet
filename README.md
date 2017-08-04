LUT Net
Proof of concept for making horribly non-differentiable things differentiable.

Ideas:

When creating the random connections, have a probability distribution for the connections.
This allows me to weight the MSBs higher than the LSBs for pixel values. Have the MSB 2 x more than (MSB-1) which is 2X more than (MSB-2) etc. See what XORnet does

Have a linebuffered pipeline architecture to do CNNs. This means sharing the LUTs for each layer of the Net in the same way that normal CNNs work. 

Generalize this to have Muxes as the fundemental unit out = Mux(In,Sel)

A LUT is then just out = Mux(In,Weights)

The Formula that I have also works for compositional Muxes (surprisingly!)

so Mux4(X00,X01,X10,X11, Sel0,Sel1) == Mux2(Mux2(X00,X01,Sel0),Mux2(X10,X11,Sel0),Sel1)

Formulat for Mux2(X0,X1,S) = (1+exp(-2/sigma))^(-1) * (X0*exp(-(S+1)^2 /(2*sigma)) + X1*exp(-(S-1)^2 /(2*sigma)))

Try to do AOT (Ahead of Time) compilation for the Muxes. This should ideally make it faster to compile (and maybe to run)

Also try out new function for MUX where instead of a gaussian or a triangle do a half gaussian/half triangle (triangle on the inside to be consistent with Linear Interpelation 


First real test:
16x16x4 -> 30

Macro and micro layers

Macro Layers do an Mk -> Mk+1
Macro layers should be such that every output bit is a function of ALL input bits

Micro Layer N->M






