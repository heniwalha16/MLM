
import { HttpClient } from '@angular/common/http';

import { movie } from '../models/movie';
import { MovieService } from '../service/movie.service';
import { Component, OnInit, Renderer2 } from '@angular/core';
import { NgbDateStruct } from '@ng-bootstrap/ng-bootstrap';

@Component({
    selector: 'app-components',
    templateUrl: './components.component.html',
    styles: [`
    ngb-progressbar {
        margin-top: 5rem;
    }
    `]
})

export class ComponentsComponent implements OnInit {
    m:movie;
    avg:number=0;
    
    ngOnInit() {
       this.m=new movie();
       
    }
    constructor( private movieService : MovieService,  private http: HttpClient){
        
    }
    
    
    
    async senddata() {
        this.movieService.postParametre(this.m).subscribe((res:any)=>{this.avg=res.average;});
        
    };
}
