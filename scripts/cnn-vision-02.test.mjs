import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const base=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../content/assets/cnn-vision-02');
const html=fs.readFileSync(base+'/demo.html','utf8');
const elements={};for(const id of ['mode','offset','offset-value','rule','input-chart','output-chart','error','maximum','fixed'])elements[id]={value:id==='mode'?'shared':'0',addEventListener(name,fn){this[name]=fn;}};
const context=vm.createContext({document:{getElementById:id=>elements[id]}});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
for(let d=0;d<16;d++){const r=vm.runInContext(`experiment(${d},true)`,context);assert.equal(r.error,0);assert.equal(r.maximum,2);}
assert.equal(vm.runInContext('experiment(8,false).error',context),4);
elements.mode.value='local';elements.offset.value='8';elements.mode.change();assert.equal(elements.error.textContent,4);
assert.equal(elements.maximum.textContent,1);
assert.equal(16*(5*5*3+1),1216);assert.equal(28*28*16*(5*5*3+1),953344);assert.equal(28*28*16*5*5*3,940800);
const w=[-1,2,-1,-1,2,-1,-1,2,-1];const patches=[[0,1,0,0,1,0,0,1,0],[0,0,0,1,1,1,0,0,0],Array(9).fill(1)];assert.deepEqual(patches.map(p=>p.reduce((v,x,i)=>v+x*w[i],0)),[6,0,0]);
for(const [,src] of html.matchAll(/src="([^"]+)"/g))assert(fs.existsSync(path.resolve(base,src)));
assert(!html.includes('<!-- demo:translation -->'));
console.log('PASS: all 16 shared shifts; unshared counterexample; actual UI handlers; parameter/MAC counts; orientation examples; image references.');
