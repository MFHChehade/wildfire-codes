clc;clear all;
A = zeros(30,186);

for i = 1:100
    A(i,:) = randperm(186);
end

B = A(:,1:4);