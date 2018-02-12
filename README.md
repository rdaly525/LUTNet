LUT Net
Proof of concept for making horribly non-differentiable things differentiable.

Ideas:

Idea for making conv better:

Instead of going from input volume to cout layers, go to N times cout layers. Do a bitwise sum of each N bits. Do Thresholding



When creating the random connections, have a probability distribution for the connections.
This allows me to weight the MSBs higher than the LSBs for pixel values. Have the MSB 2 x more than (MSB-1) which is 2X more than (MSB-2) etc. See what XORnet does

Have a linebuffered pipeline architecture to do CNNs. This means sharing the LUTs for each layer of the Net in the same way that normal CNNs work. 

Generalize this to have Muxes as the fundemental unit out = Mux(In,Sel)

A LUT is then just out = Mux(In,Weights)

The Formula that I have also works for compositional Muxes (surprisingly!)

so Mux4(X00,X01,X10,X11, Sel0,Sel1) == Mux2(Mux2(X00,X01,Sel0),Mux2(X10,X11,Sel0),Sel1)

Formula for Mux2(X0,X1,S) = (1+exp(-2/sigma))^(-1) * (X0*exp(-(S+1)^2 /(2*sigma)) + X1*exp(-(S-1)^2 /(2*sigma)))

Try to do AOT (Ahead of Time) compilation for the Muxes. This should ideally make it faster to compile (and maybe to run)

Also try out new function for MUX where instead of a gaussian or a triangle do a half gaussian/half triangle (triangle on the inside to be consistent with Linear Interpelation 

Consider using a [hyperparameter search library](http://sheffieldml.github.io/GPyOpt/)


Questions:
LUTs have N inputs. How do I do a reduction from K (>N) to 1. Basically what connections should I choose?
What shoudl the LUT function be?
  "triangle" LERP
  "gaussian"
  ??
Should there be an additional nonlinearity on the output of the LUTs?
How should I encode the inputs? Should I treat the bits as channels? Should I treat MSbs more important than LSBs?
How should I encode outputs? single bit per class is not enough information.
  Mutliple its per class, but how to design an architecture
How should I do the Loss? currently just doing L2


How to encode these functions with GPU performance in mind? (I am not doing normal convolutions anymore)




