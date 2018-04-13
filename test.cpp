#include <iostream>
#include <cstring>
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace std;

string inPath = "./wiki50k/";

extern "C"
void setInPath(char *path) {
    int len = strlen(path);
    inPath = "";
    for (int i = 0; i < len; i++)
        inPath = inPath + path[i];
    printf("Input Files Path : %s\n", inPath.c_str());
}

int relationTotal;
int entityTotal;
int testTotal, tripleTotal, trainTotal, validTotal;
float l1_filter_tot[6], r1_filter_tot[6], l1_tot[6], r1_tot[6];
float l3_filter_tot[6], r3_filter_tot[6], l3_tot[6], r3_tot[6];
float l10_filter_tot[6], r10_filter_tot[6], l10_tot[6], r10_tot[6];
float l_filter_rank[6], r_filter_rank[6], l_rank[6], r_rank[6];


struct Triple {
    int h, r, t, label;
};

struct cmp_head {
    bool operator()(const Triple &a, const Triple &b) {
        return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
    }
};

Triple *testList, *tripleList;
Triple *classList;
int *flagList;
int tripleclassTotal;
int nntotal[5];
int *head_lef, *head_rig, *tail_lef, *tail_rig;
int *head_type, *tail_type;

extern "C"
void init() {
    FILE *fin;
    int tmp, h, r, t, label;
	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%d", &relationTotal);
    fclose(fin);
	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%d", &entityTotal);
    fclose(fin);
    head_lef = (int *)calloc(relationTotal * 2, sizeof(int));
    head_rig = (int *)calloc(relationTotal * 2, sizeof(int));
    tail_lef = (int *)calloc(relationTotal * 2, sizeof(int));
    tail_rig = (int *)calloc(relationTotal * 2, sizeof(int));
    FILE* f_kb1 = fopen((inPath + "test2id_all.txt").c_str(), "r");
    FILE* f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
    FILE* f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
    tmp = fscanf(f_kb1, "%d", &testTotal);
    tmp = fscanf(f_kb2, "%d", &trainTotal);
    tmp = fscanf(f_kb3, "%d", &validTotal);
    head_type = (int *)calloc((testTotal + trainTotal + validTotal) * 2, sizeof(int));
    tail_type = (int *)calloc((testTotal + trainTotal + validTotal) * 2, sizeof(int));
    tripleTotal = testTotal + trainTotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    for (int i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%d", &label);
        tmp = fscanf(f_kb1, "%d", &h);
        tmp = fscanf(f_kb1, "%d", &t);
        tmp = fscanf(f_kb1, "%d", &r);
        label++;
        nntotal[label]++;
        testList[i].label = label;
        testList[i].h = h;
        testList[i].t = t;
        testList[i].r = r;
        tripleList[i].h = h;
        tripleList[i].t = t;
        tripleList[i].r = r;
    }
    for (int i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%d", &h);
        tmp = fscanf(f_kb2, "%d", &t);
        tmp = fscanf(f_kb2, "%d", &r);
        tripleList[i + testTotal].h = h;
        tripleList[i + testTotal].t = t;
        tripleList[i + testTotal].r = r;
    }
    for (int i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%d", &h);
        tmp = fscanf(f_kb3, "%d", &t);
        tmp = fscanf(f_kb3, "%d", &r);
        tripleList[i + testTotal + trainTotal].h = h;
        tripleList[i + testTotal + trainTotal].t = t;
        tripleList[i + testTotal + trainTotal].r = r;
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);
    sort(tripleList, tripleList + tripleTotal, cmp_head());
    
    memset(l10_filter_tot, 0, sizeof(l10_filter_tot));
    memset(r10_filter_tot, 0, sizeof(r10_filter_tot));
    memset(l10_tot, 0, sizeof(l10_tot));
    memset(r10_tot, 0, sizeof(r10_tot));
    memset(l3_filter_tot, 0, sizeof(l3_filter_tot));
    memset(r3_filter_tot, 0, sizeof(r3_filter_tot));
    memset(l3_tot, 0, sizeof(l3_tot));
    memset(r3_tot, 0, sizeof(r3_tot));
    memset(l1_filter_tot, 0, sizeof(l1_filter_tot));
    memset(r1_filter_tot, 0, sizeof(r1_filter_tot));
    memset(l1_tot, 0, sizeof(l1_tot));
    memset(r1_tot, 0, sizeof(r1_tot));

    int total_lef = 0;
    int total_rig = 0;
    FILE* f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    tmp = fscanf(f_type, "%d", &tmp);
    for (int i = 0; i < relationTotal; i++) {
        int rel, tot;
        tmp = fscanf(f_type, "%d%d", &rel, &tot);
        head_lef[rel] = total_lef;
        for (int j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%d", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%d%d", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (int j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%d", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);

}

bool find(int h, int t, int r) {
    int lef = 0;
    int rig = tripleTotal - 1;
    int mid;
    while (lef + 1 < rig) {
        int mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
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
int getTestTotal() {
	return testTotal;
}

extern "C"
int getClassTotal() {
    return tripleclassTotal;
}

extern "C"
void getClassBatch(int *ph, int *pt, int *pr, int *flag) {
    for (int i = 0; i < tripleclassTotal; i++) {
        ph[i] = classList[i].h;
        pt[i] = classList[i].t;
        pr[i] = classList[i].r;
        flag[i] = flagList[i];
    }
}

struct classTriple {
    float res;
    int flag;
};

struct classcmp {
    bool operator()(const classTriple &a, const classTriple &b) {
        return (a.res < b.res)||(a.res == b.res && a.flag > b.flag);
    }
};

extern "C"
void getClassRes(float *con, int *flag) {
    float posTot = 0;
    float negTot = 0;
    classTriple* res = (classTriple *)calloc(tripleclassTotal, sizeof(classTriple));
    for (int i = 0; i < tripleclassTotal; i++) {
        res[i].res = con[i];
        res[i].flag = flag[i];
        if (res[i].flag == 1) posTot++; else negTot++;
    }
    sort(res, res + tripleclassTotal, classcmp());
    float posS = 0;
    float negS = 0;
    float mx;
    for (int i = 0; i < tripleclassTotal; i++) {
        if (res[i].flag == 1) posS++; else negS++;
        float pre = (posS + negTot - negS)/(negTot+posTot);
        if (pre > mx) mx = pre;
    }
    printf("%f\n", mx);
    free(res);
}


int lastHead = 0;
int lastTail = 0;

extern "C"
void getHeadBatch(int *ph, int *pt, int *pr) {
	for (int i = 0; i < entityTotal; i++) {
		ph[i] = i;
		pt[i] = testList[lastHead].t;
		pr[i] = testList[lastHead].r;
	}
}

extern "C"
void getTailBatch(int *ph, int *pt, int *pr) {
	for (int i = 0; i < entityTotal; i++) {
		ph[i] = testList[lastTail].h;
		pt[i] = i;
		pr[i] = testList[lastTail].r;
	}
}

extern "C"
void testHead(float *con) {
	int h = testList[lastHead].h;
	int t = testList[lastHead].t;
	int r = testList[lastHead].r;
    int label = testList[lastHead].label;

	float minimal = con[h];
    printf("%f\n", minimal);
	int l_s = 0;
	int l_filter_s = 0;
    int l_s_constrain = 0;
    int l_filter_s_constrain = 0;
    int type_head = head_lef[r];

    for (int j = 0; j < entityTotal; j++) {
        float value = con[j];
        if (j != h && value < minimal) {
            l_s += 1;
            if (not find(j, t, r))
                l_filter_s += 1;
        }
        if (j != h) {
            while (type_head < head_rig[r] && head_type[type_head] < j) type_head++;
            if (type_head < head_rig[r] && head_type[type_head] == j) {
                    if (value < minimal) {
                        l_s_constrain += 1;
                        if (not find(j, t, r))
                            l_filter_s_constrain += 1;
                    }
            }
        }
    }

    l_filter_rank[label] += l_filter_s;
    l_rank[label] += l_s;
    l_filter_rank[5] += l_filter_s_constrain;
    l_rank[5] += l_s_constrain;
	l_filter_rank[0] += (l_filter_s+1);
	l_rank[0] += (1+l_s);

    if (l_filter_s < 10) l10_filter_tot[0] += 1;
    if (l_s < 10) l10_tot[0] += 1;
    if (l_filter_s < 3) l3_filter_tot[0] += 1;
    if (l_s < 3) l3_tot[0] += 1;
    if (l_filter_s < 1) l1_filter_tot[0] += 1;
    if (l_s < 1) l1_tot[0] += 1;

    if (l_filter_s_constrain < 10) l10_filter_tot[5] += 1;
    if (l_s_constrain < 10) l10_tot[5] += 1;
    if (l_filter_s_constrain < 3) l3_filter_tot[5] += 1;
    if (l_s_constrain < 3) l3_tot[5] += 1;
    if (l_filter_s_constrain < 1) l1_filter_tot[5] += 1;
    if (l_s_constrain < 1) l1_tot[5] += 1;

    if (l_filter_s < 10) l10_filter_tot[label] += 1;
    if (l_s < 10) l10_tot[label] += 1;
    if (l_filter_s < 3) l3_filter_tot[label] += 1;
    if (l_s < 3) l3_tot[label] += 1;
    if (l_filter_s < 1) l1_filter_tot[label] += 1;
    if (l_s < 1) l1_tot[label] += 1;
   
	lastHead++;
    printf("l_filter_s: %d\n", l_filter_s);
	printf("%f %f %f %f\n", l10_tot[0] / lastHead, l10_filter_tot[0] / lastHead, l_rank[0], l_filter_rank[0]);
}

extern "C"
void testTail(float *con) {
	int h = testList[lastTail].h;
	int t = testList[lastTail].t;
	int r = testList[lastTail].r;
    int label = testList[lastTail].label;

	float minimal = con[t];
    printf("%f\n", minimal);
	int r_s = 0;
	int r_filter_s = 0;
    int r_s_constrain = 0;
    int r_filter_s_constrain = 0;
    int type_tail = tail_lef[r];

    for (int j = 0; j < entityTotal; j++) {
        float value = con[j];
        if (j != t && value < minimal) {
            r_s += 1;
            if (not find(h, j, r))
                r_filter_s += 1;
        }
        if (j != t) {
            while (type_tail < tail_rig[r] && tail_type[type_tail] < j) type_tail++;
            if (type_tail < tail_rig[r] && tail_type[type_tail] == j) {
                if (value < minimal) {
                    r_s_constrain += 1;
                    if (not find(h, j, r))
                        r_filter_s_constrain += 1;
                }
            }
        }
    }

    r_filter_rank[label] += r_filter_s;
    r_rank[label] += r_s;
    r_filter_rank[5] += r_filter_s_constrain;
    r_rank[5] += r_s_constrain;
	r_filter_rank[0] += (1+r_filter_s);
	r_rank[0] += (1+r_s);

	if (r_filter_s < 10) r10_filter_tot[0] += 1;
	if (r_s < 10) r10_tot[0] += 1;
    if (r_filter_s < 3) r3_filter_tot[0] += 1;
    if (r_s < 3) r3_tot[0] += 1;
    if (r_filter_s < 1) r1_filter_tot[0] += 1;
    if (r_s < 1) r1_tot[0] += 1;

    if (r_filter_s_constrain < 10) r10_filter_tot[5] += 1;
    if (r_s_constrain < 10) r10_tot[5] += 1;
    if (r_filter_s_constrain < 3) r3_filter_tot[5] += 1;
    if (r_s_constrain < 3) r3_tot[5] += 1;
    if (r_filter_s_constrain < 1) r1_filter_tot[5] += 1;
    if (r_s_constrain < 1) r1_tot[5] += 1;

    if (r_filter_s < 10) r10_filter_tot[label] += 1;
    if (r_s < 10) r10_tot[label] += 1;
    if (r_filter_s < 3) r3_filter_tot[label] += 1;
    if (r_s < 3) r3_tot[label] += 1;
    if (r_filter_s < 1) r1_filter_tot[label] += 1;
    if (r_s < 1) r1_tot[label] += 1;

	lastTail++;
    printf("r_filter_s: %d\n", r_filter_s);
	printf("%f %f %f %f\n", r10_tot[0] /lastTail, r10_filter_tot[0] /lastTail, r_rank[0], r_filter_rank[0]);
}

extern "C"
void test() {
    printf("results:\n");
    for (int i = 0; i <=0; i++) {
    	printf("left %f %f %f %f \n", l_rank[i] / testTotal, l10_tot[i] / testTotal, l3_tot[i] / testTotal, l1_tot[i] / testTotal);
        printf("left(filter) %f %f %f %f \n", l_filter_rank[i] / testTotal, l10_filter_tot[i] / testTotal, l3_filter_tot[i] / testTotal, l1_filter_tot[i] / testTotal);
        printf("right %f %f %f %f \n", r_rank[i] / testTotal, r10_tot[i] / testTotal, r3_tot[i] / testTotal, r1_tot[i] / testTotal);
        printf("right(filter) %f %f %f %f\n", r_filter_rank[i] / testTotal, r10_filter_tot[i] / testTotal, r3_filter_tot[i] / testTotal, r1_filter_tot[i] / testTotal);
    }
    printf("results (type constraints):\n");
    for (int i = 5; i <=5; i++) {
        printf("left %f %f %f %f \n", l_rank[i] / testTotal, l10_tot[i] / testTotal, l3_tot[i] / testTotal, l1_tot[i] / testTotal);
        printf("left(filter) %f %f %f %f \n", l_filter_rank[i] / testTotal, l10_filter_tot[i] / testTotal, l3_filter_tot[i] / testTotal, l1_filter_tot[i] / testTotal);
        printf("right %f %f %f %f \n", r_rank[i] / testTotal, r10_tot[i] / testTotal, r3_tot[i] / testTotal, r1_tot[i] / testTotal);
        printf("right(filter) %f %f %f %f\n", r_filter_rank[i] / testTotal, r10_filter_tot[i] / testTotal, r3_filter_tot[i] / testTotal, r1_filter_tot[i] / testTotal);
    }
    for (int i = 1; i <= 4; i++) {
        if (i == 1)
            printf("results (1-1):\n");
        if (i == 2)
            printf("results (1-n):\n");
        if (i == 3)
            printf("results (n-1):\n");
        if (i == 4)
            printf("results (n-n):\n");
        printf("left %f %f %f %f \n", l_rank[i] / nntotal[i], l10_tot[i] / nntotal[i], l3_tot[i] / nntotal[i], l1_tot[i] / nntotal[i]);
        printf("left(filter) %f %f %f %f \n", l_filter_rank[i] / nntotal[i], l10_filter_tot[i] / nntotal[i], l3_filter_tot[i] / nntotal[i], l1_filter_tot[i] / nntotal[i]);
        printf("right %f %f %f %f \n", r_rank[i] / nntotal[i], r10_tot[i] / nntotal[i], r3_tot[i] / nntotal[i], r1_tot[i] / nntotal[i]);
        printf("right(filter) %f %f %f %f\n", r_filter_rank[i] / nntotal[i], r10_filter_tot[i] / nntotal[i], r3_filter_tot[i] / nntotal[i], r1_filter_tot[i] / nntotal[i]);
    }
}


int main() {
    init();
    return 0;
}
