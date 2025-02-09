clear all
close all
%% Preparing an exam
env(1,1).next_state=[1]; %Starting State 1: Facebook
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

%% Start to Learn with TD Reinforcement Learning
Q=zeros(4,2);
final_point=0; %flag = 1 if the terminal state (5) is reached
Discount=1;
rng(1);
alpha=0.1

Nsim=10000;
epsilon=1;% epsilon-gredy with rate epsilon/sqrt(ii), ii=#simulation
for ii=1:Nsim
    final_point=0;
    i=1;
    actions=[];
    rewards=[];
    states=1;
    while(final_point==0)
        if (rand<epsilon/sqrt(ii))
            actions(i)=(1+round(rand));
        else
            [~,actions(i)]=max(Q(states(i),:));
        end
        
        [states(i+1),rewards(i)]=next_state(env, states(i),actions(i));
        if  (states(i+1)==5)
            Q(states(i),actions(i))=Q(states(i),actions(i))+...
                alpha*(rewards(i)-Q(states(i),actions(i)));
            final_point=1;
        else
            Q(states(i),actions(i))=Q(states(i),actions(i))+...
         alpha*(rewards(i)+Discount*max(Q(states(i+1),:))-Q(states(i),actions(i)));
        end
        i=i+1;
    end

end
Q
V=max(Q,[],2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
