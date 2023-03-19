import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { movie } from '../models/movie';
@Injectable({
  providedIn: 'root'
})
export class MovieService {

  url='http://127.0.0.1:8000/predict/'
  constructor(private http:HttpClient) { }
  postParametre(m:movie){
  console.log(m.title);
  return this.http.post('http://127.0.0.1:8000/predict/',m.title);
  }
  getAverage(){
    return this.http.get("http://127.0.0.1:8000/predict/");
  }
}
