#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>

using namespace std;

string inPath = "./data/";
int bernFlag = 0;

extern "C"
void setInPath(char *path) {
    int len = strlen(path);
    inPath = "";
    for (int i = 0; i < len; i++)
        inPath = inPath + path[i];
    printf("Input Files Path : %s\n", inPath.c_str());
}

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
};

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

struct cmp_list {
	int minimal(int a,int b) {
		if (a > b) return b;
		return a;
	}
	bool operator()(const Triple &a, const Triple &b) {
		return (minimal(a.h, a.t) > minimal(b.h, b.t));
	}
};

Triple *trainHead, *trainTail, *trainList;
int relationTotal, entityTotal, tripleTotal;
int *freqRel, *freqEnt;
float *left_mean, *right_mean;

extern "C"
void init() {

	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

	freqRel = (int *)calloc(relationTotal, sizeof(int));
	
	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

	freqEnt = (int *)calloc(entityTotal, sizeof(int));
	
	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	tripleTotal = 0;
	while (fscanf(fin, "%d", &trainList[tripleTotal].h) == 1) {
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].t);
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].r);
		freqEnt[trainList[tripleTotal].t]++;
		freqEnt[trainList[tripleTotal].h]++;
		freqRel[trainList[tripleTotal].r]++;
		trainHead[tripleTotal].h = trainList[tripleTotal].h;
		trainHead[tripleTotal].t = trainList[tripleTotal].t;
		trainHead[tripleTotal].r = trainList[tripleTotal].r;
		trainTail[tripleTotal].h = trainList[tripleTotal].h;
		trainTail[tripleTotal].t = trainList[tripleTotal].t;
		trainTail[tripleTotal].r = trainList[tripleTotal].r;
		tripleTotal++;
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());

	lefHead = (int *)calloc(entityTotal, sizeof(int));
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(rigHead));
	memset(rigTail, -1, sizeof(rigTail));
	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	left_mean = (float *)calloc(relationTotal,sizeof(float));
	right_mean = (float *)calloc(relationTotal,sizeof(float));
	for (int i = 0; i < entityTotal; i++) {
		for (int j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (int j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (int i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
}

extern "C"
int getEntityTotal() {
	return entityTotal;
}

extern "C"
int getRelationTotal() {
	return relationTotal;
}

extern "C"
int getTripleTotal() {
	return tripleTotal;
}

extern "C"
void setBernFlag(int flag = 0) {
	bernFlag = flag;
};

// unsigned long long *next_random;
unsigned long long next_random = 3;

unsigned long long randd(int id) {
	next_random = next_random * (unsigned long long)25214903917 + 11;
	return next_random;
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

extern "C"
void getBatch(int *ph, int *pt, int *pr, int *nh, int *nt, int *nr, int batchSize, int id = 0) {
	int i, j;
	float prob;
	for (int batch = 0; batch < batchSize; batch++) {
		i = rand_max(id, tripleTotal);
		if (bernFlag)
			prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
		else
			prob = 500;
		if (randd(id) % 1000 < prob) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			ph[batch] = trainList[i].h;
			pt[batch] = trainList[i].t;
			pr[batch] = trainList[i].r;
			nh[batch] = trainList[i].h;
			nt[batch] = j;
			nr[batch] = trainList[i].r;
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			ph[batch] = trainList[i].h;
			pt[batch] = trainList[i].t;
			pr[batch] = trainList[i].r;
			nh[batch] = j;
			nt[batch] = trainList[i].t;
			nr[batch] = trainList[i].r;
		}
	}
}
