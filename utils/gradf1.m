function gradf = gradf1(x,y2,a,gama,tau1,tau2,tau3)
gradf =tau1*(gama-exp(y2-x)) + tau2*(dtd(x)) + tau3*(x-a);
end