function [Q,N_Q] = update_Q(Q,N_Q,actions,states,rewards,Discount)
Dim=length(rewards); %number of steps to arrive to final state 5
for i=1:Dim
    N_Q(states(i),actions(i)) = ...
        N_Q(states(i),actions(i))+1;
    sum=0;
    for j=i:Dim
        sum=sum+Discount^(j-i)*rewards(j);
    end
    Q(states(i),actions(i))= Q(states(i),actions(i))+...
        1/N_Q(states(i),actions(i))*(sum -Q(states(i),actions(i)));

end


end

