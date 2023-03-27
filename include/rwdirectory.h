/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   rwdirectory.h
 * Author: alireza
 *
 * Created on April 9, 2019, 10:56 AM
 */

#ifndef RWDIRECTORY_H
#define RWDIRECTORY_H
/*#include <bits/stdc++.h> */
/*#include<iostream>*/
/*#include <string>*/
/*#include <dirent.h>*/
/*#include<vector>*/
#include <sys/stat.h> 
/*#include <sys/types.h> */
//#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include<cstring>
using namespace std;

namespace directory {

    vector<string> readingFileOfDir(string adr, string ext) {
        DIR *dir;
        struct dirent *file;
        dir = opendir(adr.c_str());
        vector <string> fileName;
        
        //int idir = 0;
        string temp;
        while ((file = readdir(dir)) != NULL) {
            temp = file->d_name;
            if (temp.find(ext) != string::npos) {
                fileName.push_back(temp);
                //cout<<fileName[idir]<<endl;               
            }
        }
        closedir(dir);
        int size=fileName.size();
        printf("# %d %s files have been read:\n", size,ext.c_str() );
        if(size>4){
        printf("1:%s\n", fileName[0].c_str());
        printf("2:%s\n", fileName[1].c_str());
        printf("3:%s\n", fileName[2].c_str());
        printf(".\n.\n.\n");
        printf("%d:%s\n", size, fileName[size-1].c_str());
        }else{
        for(int i=0;i<size; i++){
        printf("%d:%s\n", i+1, fileName[i].c_str());
        }
        }
/*        for (string &s : fileName) {*/
/*            std::cout << s << endl;*/
/*        }*/
        return fileName;
    }

    string makeFolder(string adr, string name) {
        DIR *dir;
        //struct dirent *file;
        string temp = adr + "/" + name;
        dir = opendir(temp.c_str());
        if (dir) {
            printf( "the directory already existed\t%d\t%s.\n",__LINE__,__FUNCTION__);
            return temp+"/";
        } else {
            int status = 0;
            status = mkdir(temp.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            printf("# the following directory has been made:%s\n#%s\n",__FUNCTION__,temp.c_str());
            if (-1 == status) {
                printf("Error creating directory!n");
                exit(1);

            }
            return temp+"/";
        }
    }
    inline vector<string> tokenize(string name, string delim){
    	char intermediate[name.size()+1];
    	strcpy(intermediate, name.c_str());
    	char* token=strtok(intermediate, delim.c_str());
    	vector<string> splited_name;
    	while( token != NULL ) {
    		splited_name.push_back(token);
    		token = strtok(NULL,delim.c_str());
    	}
    	return splited_name;
    }
	string make_nested_folder(string addr){
		DIR *dir;
		dir = opendir(addr.c_str());
		if(dir){
			printf("the directory already existed!%d\t%s\n",__LINE__, __FUNCTION__);
			closedir(dir);
			return addr;
		}else{
		vector<string> a=directory::tokenize(addr,"/");
		int o=a.size();
		string temp=a[0];
		for (int i=0;i<o-1;i++){
		makeFolder(temp, a[i+1]);
		temp=temp+"/"+a[i+1];
		}
		return temp;}
}
    void writing_the_results(double *trace, string dir, string name, size_t size){
        string totalName = dir + name + ".bin";
        ofstream res(totalName, ios::out|ios::binary);
        res.precision(12);
        if (!res) {
            cout << "Error happened in opening the file ";
        }
        res.write(reinterpret_cast<const char*> (trace), size);
        res.close();
    }

};


#endif /* RWDIRECTORY_H */

