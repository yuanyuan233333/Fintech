clear all
close all
%% Preparing an exam
env(1,1).next_state=[1];  %Starting State 1: Facebook
env(1,1).reward=[-1];
env(1,1).prob=[1];
env(1,2).next_state=[2];
env(1,2).reward=[0];
env(1,2).prob=[1];

env(2,1).next_state=[1];
env(2,1).reward=[-1];
env(2,1).prob=[1];
env(2,2).next_state=[3];
env(2,2).reward=[-2];
env(2,2).prob=[1];

env(3,1).next_state=[5];
env(3,1).reward=[0];
env(3,1).prob=[1];
env(3,2).next_state=[4];
env(3,2).reward=[-2];
env(3,2).prob=[1];

env(4,1).next_state=[5];
env(4,1).reward=[10];
env(4,1).prob=[1];
env(4,2).next_state=[2,3];
env(4,2).reward=[1];
env(4,2).prob=[0.4,0.6];

Q=zeros(4,2); % Q(s,a) --> action value function (state, action)
Q_N=zeros(4,2); % number of times (state,action) is considered
%% One (random) single Exploration
final_point=0; %flag = 1 if the terminal state (5) is reached
states(1)= 1; % starting state
i=1;
Discount=1;
rng(1);
while(final_point==0)
    actions(i)=1+round(rand);
    [states(i+1),rewards(i)]=next_state(env, states(i),actions(i));
    i=i+1;
    final_point=(states(i)==5);
end
[Q,Q_N]=update_Q(Q,Q_N,actions,states(1:end-1),rewards,Discount);
%% Start to Learn with MC Reinforcement Learning
Nsim=1000000;
epsilon=1; % epsilon-gredy with rate epsilon/sqrt(ii), ii=#simulation
for ii=1:Nsim
    final_point=0;
    i=1;
    actions=[];
    rewards=[];
    states=1;
    while(final_point==0)
        if (rand<epsilon/sqrt(ii))
            actions(i)=1+round(rand);
        else
            [~,actions(i)]=max(Q(states(i),:));            
        end
        [states(i+1),rewards(i)]=next_state(env, states(i),actions(i));
        i=i+1;
        final_point=(states(i)==5);
    end
    [Q,Q_N]=update_Q(Q,Q_N,actions,states(1:end-1),rewards,Discount);

end
Q
V=max(Q,[],2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
