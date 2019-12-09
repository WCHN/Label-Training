clear Ws
n=0;
for i=1:2
    for j=1:2
        for k=1:2
            a{k}=k;
            b{j}=j;
            c{i}=i;
           
     n=n+1;   
        end
    end
end




 
M = numel(data.code);
N = 0;

for n=1:numel(data.dat)
    N = N+prod(data.dat(n).jitter*2+1);
end
     
 h=prod(data.dat(1).jitter*2+1);%h=27
 h1=data.dat(1).jitter*2+1; %h1=[3 3 3]
 b=numel(data.dat); % 15
 

 

    
 
 
 
 
 
 
 
 
 