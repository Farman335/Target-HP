clc;
close all;
clear all;
M1=zeros(1056,20);
M2=zeros(1056,20);

lambda=2; %lambda can be 0,1,2
for t=1:1056
    t
    PR=csvread(['pssm' num2str(t),'.xls']);
    P=PR(:,1:20);
    [L,n]=size(P);
    fL=round(L/2);
  for i=1:n
       for j=1:(fL-lambda)
             M1(t,i)=M1(t,i)+(P(j,i)-P((j+lambda),i))^2/(fL-lambda);
%            A1(t,i)=A1(t,i)+P(j,i)/fL;
       end
  end
   
  for i=1:n
       for j=(fL+1):(L-lambda)
           M2(t,i)=M2(t,i)+(P(j,i)-P((j+lambda),i))^2/(L-fL-lambda);
%           A2(t,i)=A2(t,i)+P(j,i)/(L-fL-lambda);
       end
  end
end
Sg2_PsePSSM_lg2_1056_org=[M1 M2];
 save Sg2_PsePSSM_lg2_1056_org

