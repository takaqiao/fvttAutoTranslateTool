/** Aid owns its action options, native Performance statistic and outcome notes. */
export async function rollNativeUpliftingAid({game,actor,target,dc,options,event,callback}){
 const aid=game.pf2e.actions.get('aid');
 if(typeof aid?.use!=='function')throw Error('原生协助动作不可用。');
 const results=await aid.use({actors:actor,target,statistic:'performance',difficultyClass:dc,rollOptions:[...options],event,message:{create:false}});
 if(!results.length)return null;
 if(results.length!==1||results[0].actor!==actor||!results[0].roll||!results[0].message)throw Error('原生协助没有返回唯一的本次检定。');
 const {roll,outcome,message}=results[0];await callback(roll,outcome,message);return roll;
}
