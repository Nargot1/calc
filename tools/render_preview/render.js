// usage: node render.js <dir> <glb> <outprefix> [flat]
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const [,, dir, glb, outp, flat='0'] = process.argv;
const views = [['iso',40,35,1.45],['iso2',215,30,1.45],['low',125,14,1.5],['top',0.0001,89,1.25]];
const mime = {'.html':'text/html','.glb':'model/gltf-binary','.js':'text/javascript'};
const srv = http.createServer((req,res)=>{ const f = path.join(dir, decodeURIComponent(req.url.split('?')[0])); fs.readFile(f,(e,d)=>{ if(e){res.writeHead(404);res.end();return;} res.writeHead(200,{'Content-Type':mime[path.extname(f)]||'application/octet-stream'}); res.end(d);});});
srv.listen(0, '127.0.0.1', async () => {
  const port = srv.address().port;
  const b = await chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
  const pg = await b.newPage({viewport:{width:1600,height:1000}});
  pg.on('console', m => { if (m.type()==='error'||m.type()==='warning') console.log('console:', m.text()); });
  for (const [name, az, el, d] of views) {
    await pg.goto(`http://127.0.0.1:${port}/viewer.html?src=${glb}&az=${az}&el=${el}&d=${d}&flat=${flat}`);
    await pg.waitForFunction('window.__done === true', null, {timeout: 240000});
    const err = await pg.evaluate('window.__err || null'); if (err) console.log('load error', err);
    await pg.screenshot({path: `${outp}_${name}.png`}); console.log('saved', `${outp}_${name}.png`);
  }
  await b.close(); srv.close();
});
