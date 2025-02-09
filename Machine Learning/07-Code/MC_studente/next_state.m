function [Next_state, reward]= next_state(env,state, action)
reward=env(state,action).reward;
if size(env(state,action).prob,2)==1
    Next_state =env(state,action).next_state;
else
    if rand<env(state,action).prob(1)
        Next_state= env(state,action).next_state(1);
    else
        Next_state= env(state,action).next_state(2);
    end
end
end