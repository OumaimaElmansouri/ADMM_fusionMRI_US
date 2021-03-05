function f = f1(x,y2,a,gama,tau1,tau2,tau3)
f =tau1*sum(sum(gama*exp(y2-x)-(y2-x))) + tau2*(norm(d1(x))^2) + tau3*norm(x-a);
end