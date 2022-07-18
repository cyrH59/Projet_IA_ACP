

% clc;
% clear all;
close all;

%% Data extraction
% Training set
adr = './training1/';
fld = dir(adr);
nb_elt = length(fld);
% Data matrix containing the training images in its columns 
data_trn = []; 
% Vector containing the class of each training image
lb_trn = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_trn = [lb_trn ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_trn = [data_trn img(:)];
    end
end
% Size of the training set
[P,N] = size(data_trn);

% calcul valeurs propres et vecteurs propres
xm=sum(data_trn,2)/N; %represente la moyenne pour chaque vecteur


X= (1/sqrt(N))*(data_trn-xm);  % permet de normaliser nos vecteurs images
XXt=X'*X; % matrice de gram XXt
[V,D]=svd(XXt); % Vecteurs propres et valeurs propres associées
Dbis=diag(D); % Valeurs propres de XXt
U=X*V*(V'*(X')*X*V)^(-1/2); 


%% representations des n eigfaces : 



h=192;
la=168;
%% Projection : 
%Q4 : 
Seuil=0.9; % Seuil defini
kl=zeros(1,size(D,1)); % ratio 
for k=1:length(kl)
    kl(k)=sum(Dbis(1:k))/sum(Dbis);
end
figure;
plot(1:size(D,1),kl)
title("kl(l)");

i=1;
while(kl(i)<Seuil)
    i=i+1;
end
disp("Le seuil est dépassé pour l= "+i );
%Q3 :
l=i; % Dimension du sous espace vectoriel [1,10,20,35,50,60]
U_=U(:,1:l);
inter=1:10:N;


proj=U_*(U_'*data_trn(:,inter)); %permet de représenter certaines images projetées 
projall=U_*(U_'*data_trn(:,1:N)); %permet de projeter l'ensemble des images



% Classes contained in the training set
[~,I]=sort(lb_trn);
data_trn = data_trn(:,I);
[cls_trn,bd,~] = unique(lb_trn);
Nc = length(cls_trn); 
% Number of training images in each class
size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1]; 
% Display the database
F = zeros(192*Nc,168*max(size_cls_trn));

for i=1:Nc
    for j=1:size_cls_trn(i)
          pos = sum(size_cls_trn(1:i-1))+j;
          F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(U(:,pos),[192,168]);
%           if(i==1)
%               tabimg(:,:,:)=reshape(projall(:,inter)+xm,[192,168]);
%           end
    end
end
% tabimg(:,:,nb)=reshape(projall(:,1)+xm,[192,168]);
% nb=nb+1;


figure;
imshow(abs(F));
colormap(gray);
title("Images issues de la base Training 1")

axis off;

%% Implementation de V(x) :
wtrain=zeros(l,N); %calcul w(x) pour chaque xi
for i=1:N
    for j=1:l
        dtmp=data_trn(:,i);
        utmp=U_(:,j);
        wtrain(j,i)=(dtmp'-xm')*utmp;
    end
end

% Argmax : 

k=3; %nombre de plus proche voisin à determiner
vx=zeros(k,N); %represente la fonction V(x) qui contient pour chaque element les k plus proches voisins
for o=1:N % pour faire l'estimation pour chaque xi
    normes=zeros(1,N); 
    for t=1:N
        normes(1,t)=norm(wtrain(:,o)-wtrain(:,t));
    end
    normes(1,o)=NaN;
    for u=1:k
        [~, indicetmp]=min(normes);
        normes(1,indicetmp)=NaN; %on efface la valeur pour déterminer le prochain min
        vx(u,o)=indicetmp; 
    end
    

end

classev=fix((vx-1)/size_cls_trn(1,1))+1; %permet d'avoir la classe de chaque image associé au indice relevé
nboccurence=zeros(size(size_cls_trn,1),N);  %permet d'avoir le nombre d'occurence par classe 

for p=1:N
    for j=1:k
        nboccurence(classev(j,p),p)=nboccurence(classev(j,p),p)+1;
    end
end

[~,classef]=max(nboccurence); %donne la classe finale 

%% matrice de confusion : 

classeinitiale=zeros(1,N);%classe initiale
for s=1:N
    classeinitiale(s)=fix((s-1)/size_cls_trn(1,1))+1;
end

[Cknn,errateknn]=confmat(classeinitiale',classef'); %% permet d'avoir la matrice confmat ainsi que le taux d'erreur


%% classificateur gaussien : 


%% Calcul de la moyenne : 
moyennetrain=zeros(l,6); %  moyenne de chaque classe
val=0;
lval=1;
somme=zeros(l,1);
for m=1:N
    if (val ~= size_cls_trn(1,1))
        val=val+1;
        somme(:,1)=somme(:,1)+wtrain(:,m);
    end
    if (val == size_cls_trn(1,1))
        val=0;
        somme(:,1)=somme(:,1)+wtrain(:,m);
        moyennetrain(:,lval)=somme/size_cls_trn(1,1);
        somme=zeros(l,1);
        lval=lval+1;

    end
end

%% Calcul de la matrice de covariance : 
% Deux versions, la première partie non commenté correspond à la formule du
% sujet alors que la deuxième partie commentée correspond à un calcul d'une
% matrice de covariance pour chaque classe.
matcovtrain=zeros(l,l); % matrice de covariance
for j=1:Nc
    for i=1:N
        matcovtrain=matcovtrain+(wtrain(:,i)-moyennetrain(:,j))*(wtrain(:,i)-moyennetrain(:,j))';
    end
end

% Q6 calcul de differentes matrices de covariance pour chaque classe 
% matcovtrain=zeros(l,l,Nc); % matrice de covariance
% 
% for j=1:Nc
%     for i=1:size_cls_trn(1,1)
%         matcovtrain(:,:,j)=matcovtrain(:,:,j)+(wtrain(:,(j-1)*size_cls_trn(1,1)+i)-moyennetrain(:,j))*(wtrain(:,(j-1)*size_cls_trn(1,1)+i)-moyennetrain(:,j))';
% 
%     end
% end


%% Representation composantes principales : 
% wbis=abs(wtrain);
% %(1,2) :
% subplot(2,2,1)
% hold on;
% for i=1:(1+2*size_cls_trn(1,1)+size_cls_trn(1,1))
%     if(fix((i-1)/size_cls_trn(1,1))==0)
%         plot(wbis(1,i),wbis(2,i),'r*');
%     end
%     if(fix((i-1)/size_cls_trn(1,1))==1)
%         plot(wbis(1,i),wbis(2,i),'g*');
%     end
%     if(fix((i-1)/size_cls_trn(1,1))==2)
%         plot(wbis(1,i),wbis(2,i),'b*');
%     end
% end
% title("representation de (1,2)")
% xlabel("composante 1");
% ylabel("composante 2");
% 
% % (2,3) :
% subplot(2,2,2)
% hold on;
% for i=1:(1+2*size_cls_trn(1,1)+size_cls_trn(1,1))
%     if(fix((i-1)/size_cls_trn(1,1))==0)
%         plot(wbis(2,i),wbis(3,i),'r*');
%     end
%     if(fix((i-1)/size_cls_trn(1,1))==1)
%         plot(wbis(2,i),wbis(3,i),'g*');
%     end
%     if(fix((i-1)/size_cls_trn(1,1))==2)
%         plot(wbis(2,i),wbis(3,i),'b*');
%     end
% end
% title("representation de (2,3)")
% xlabel("composante 2");
% ylabel("composante 3");
% 
% % (3,4) :
% subplot(2,2,3)
% hold on;
% for i=1:(1+2*size_cls_trn(1,1)+size_cls_trn(1,1))
%     if(fix((i-1)/size_cls_trn(1,1))==0)
%         plot(wbis(3,i),wbis(4,i),'r*');
%     end
%     if(fix((i-1)/size_cls_trn(1,1))==1)
%         plot(wbis(3,i),wbis(4,i),'g*');
%     end
%     if(fix((i-1)/size_cls_trn(1,1))==2)
%         plot(wbis(3,i),wbis(4,i),'b*');
%     end
% end
% title("representation de (3,4)")
% xlabel("composante 3");
% ylabel("composante 4");
% % (4,5) :
% subplot(2,2,4)
% hold on;
% for i=1:(1+2*size_cls_trn(1,1)+size_cls_trn(1,1))
%     if(fix((i-1)/size_cls_trn(1,1))==0)
%         plot(wbis(4,i),wbis(5,i),'r*');
%     end
%     if(fix((i-1)/size_cls_trn(1,1))==1)
%         plot(wbis(4,i),wbis(5,i),'g*');
%     end
%     if(fix((i-1)/size_cls_trn(1,1))==2)
%         plot(wbis(4,i),wbis(5,i),'b*');
%     end
% end
% title("representation de (4,5)")
% xlabel("composante 4");
% ylabel("composante 5");

%% Implementation classifieur gaussien : 

phix=zeros(1,N);

for k=1:N
    tabnorm=zeros(1,6);
    for j=1:Nc
        tabnorm(1,j)=(norm(matcovtrain^(-1/2)*(wtrain(:,k)-moyennetrain(:,j))))^2;
    end
    [~,indicetmp]=min(tabnorm);
    phix(1,k)=indicetmp;
end

% %Q6 : calcul de norme avec plusieurs matrices de covariance 
% for k=1:N
%     tabnorm=zeros(1,6);
%     for j=1:Nc
%         tabnorm(1,j)=(norm(matcovtrain(:,:,j)^(-1/2)*(wtrain(:,k)-moyennetrain(:,j))))^2;
%     end
%     [~,indicetmp]=min(tabnorm);
%     phix(1,k)=indicetmp;
% end
[Cgaussien,errategaussien]=confmat(classeinitiale',phix'); %% permet d'avoir la matrice confmat ainsi que le taux d'erreur




%% Partie test :

% Training set
adrtest = './test1/';
fldtest = dir(adrtest);
nb_elttest = length(fldtest);
% Data matrix containing the training images in its columns 
data_trntest = []; 
% Vector containing the class of each training image
lb_trntest = []; 
for i=1:nb_elttest
    if fldtest(i).isdir == false
        lb_trntest = [lb_trntest ; str2num(fldtest(i).name(6:7))];
        imgtest = double(imread([adrtest fldtest(i).name]));
        data_trntest = [data_trntest imgtest(:)];
    end
end
% Size of the test set
[Ptest,Ntest] = size(data_trntest);

% calcul valeurs propres et vecteurs propres
xmtest=sum(data_trntest,2)/Ntest; %represente la moyenne pour chaque vecteur

%% representations des n eigfaces : 



htest=192;
latest=168;
%% Projection : 
% Classes contained in the training set
[~,Itest]=sort(lb_trntest);
data_trntest = data_trntest(:,Itest);
[cls_trntest,bdtest,~] = unique(lb_trntest);
Nctest = length(cls_trntest); 
% Number of training images in each class
size_cls_trntest = [bdtest(2:Nctest)-bdtest(1:Nctest-1);Ntest-bdtest(Nctest)+1]; 
% Display the database
Ftest = zeros(192*Nctest,168*max(size_cls_trntest));

%% classificateur RNN : 

% Implementation de V(x) :
wtest=zeros(l,Ntest); %calcul w(x) pour chaque données
for i=1:Ntest
    for j=1:l
        dtmptest=data_trntest(:,i);
        utmptest=U(:,j);
        wtest(j,i)=(dtmptest'-xmtest')*utmptest;
    end
end

ktest=5; %nombre de plus proche voisin à determiner
vxtest=zeros(ktest,Ntest); %represente la fonction V(x) qui contient pour chaque element les k plus proches voisins
for o=1:Ntest % pour faire l'estimation pour chaque xi
    normestest=zeros(1,N); 
    for t=1:N
        normestest(1,t)=norm(wtest(:,o)-wtrain(:,t));
    end

%     normestest(1,o)=NaN;
    for u=1:ktest
        [~, indicetmptest]=min(normestest);
        normestest(1,indicetmptest)=NaN; %on efface la valeur pour déterminer le prochain min
        vxtest(u,o)=indicetmptest; 
    end
       

end

classevtest=fix(((vxtest)-1)/size_cls_trn(1,1))+1; %permet d'avoir la classe de chaque image associé au indice relevé
nboccurencetest=zeros(size(size_cls_trntest,1),Ntest);  %permet d'avoir le nombre d'occurence par classe 

for p=1:Ntest
    for j=1:ktest
        nboccurencetest(classevtest(j,p),p)=nboccurencetest(classevtest(j,p),p)+1;
    end
end

[~,classeftest]=max(nboccurencetest); %donne la classe finale 

classeinitialetest=zeros(1,Ntest);%classe initiale
for s=1:Ntest
    classeinitialetest(s)=fix((s-1)/size_cls_trntest(1,1))+1;
end

[Cknntest,errateknntest]=confmat(classeinitialetest',classeftest'); %% permet d'avoir la matrice confmat ainsi que le taux d'erreur

%% Classificateur Gaussien : 

phixtest=zeros(1,Ntest);

for k=1:Ntest
    tabnormtest=zeros(1,6);
    for j=1:Nctest
        tabnormtest(1,j)=(norm(matcovtrain^(-1/2)*(wtest(:,k)-moyennetrain(:,j))))^2;
    end
    [~,indicetmptest]=min(tabnormtest);
    phixtest(1,k)=indicetmptest;
end

%Q6, utilisation différentes matrices de covariances :
% for k=1:Ntest
%     tabnormtest=zeros(1,6);
%     for j=1:Nctest
%         tabnormtest(1,j)=(norm(matcovtrain(:,:,j)^(-1/2)*(wtest(:,k)-moyennetrain(:,j))))^2;
%     end
%     [~,indicetmptest]=min(tabnormtest);
%     phixtest(1,k)=indicetmptest;
% end


[Cgaussientest,errategaussientest]=confmat(classeinitialetest',phixtest'); %% permet d'avoir la matrice confmat ainsi que le taux d'erreur


%% Résultats : 

erreurknn=[0 0 0.0606 0.3148 0.6444 0.7917];
erreurgaussien= [0 0 0.0152 0.0370 0.4111 0.7500];
erreurgaussienq6=[0 0 0 0.14 0.55 0.76];

figure;
hold on;
plot([1 2 3 4 5 6], erreurknn*100,'b');
plot([1 2 3 4 5 6], erreurgaussien*100,'r');

title('Comparaison entre estimateur gaussien et knn');
xlabel('numero test');
ylabel('Pourcentage erreur')
legend('Estimateur KNN','Estimateur Gaussien','Location','NorthWest' );

erreurknn5= [0 0 0.0758 0.2593 0.6444 0.8194];
erreurknn3=  [0 0 0.0606 0.3148 0.6444 0.7917];
erreurknn1= [0 0 0.0455 0.2593 0.5778 0.7222];

figure;
hold on;
plot([1 2 3 4 5 6], erreurknn1*100,'b-o');
plot([1 2 3 4 5 6], erreurknn3*100,'r-s');
plot([1 2 3 4 5 6], erreurknn5*100,'g-x');
plot([1 2 3 4 5 6], erreurgaussien*100,'black-p');

title('Comparaison Méthode KNN et Méthode Gaussienne');
xlabel('Numero test');
ylabel('Pourcentage erreur')
legend('KNN (k=1)','KNN (k=3)','KNN (k=5)','Gaussienne','Location','NorthWest');

% ltab=[1 10 20 35 50 60]
% figure;
% sgtitle("1 image de Training 1 pour different l ")
% for k=1:6 
% subplot(3,3,k)
% imshow(uint8(tabimg(:,:,k)))
% title("l="+ltab(k));

figure; 
hold on 

plot([1 2 3 4 5 6], erreurgaussien*100,'r-x');
plot([1 2 3 4 5 6], erreurgaussienq6*100,'b-o');
title('Méthode gaussien');
xlabel('Numero test');
ylabel('Pourcentage erreur')
legend('Calcul classique de la matrice gaussienne','Calcul matrice gaussienne différentes','Location','NorthWest');


% end

