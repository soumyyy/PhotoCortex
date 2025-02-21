import Image from 'next/image';
import { Person } from '@/types';

interface PeopleGridProps {
  uniquePeople: Person[];
}

export function PeopleGrid({ uniquePeople }: PeopleGridProps) {
  // Separate people with multiple instances from those with single instances
  const multiplePeople = uniquePeople.filter(person => person.instances_count > 1);
  const singlePeople = uniquePeople.filter(person => person.instances_count === 1);

  return (
    <div className="space-y-8">
      {/* Multiple Instances Section */}
      {multiplePeople.length > 0 && (
        <div>
          <h2 className="text-2xl font-semibold mb-4">People with Multiple Appearances ({multiplePeople.length})</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {multiplePeople.map((person) => (
              <div key={person.person_id} className="bg-gray-800 rounded-lg p-4">
                <div className="relative aspect-square mb-4">
                  <Image
                    src={person.representative_face.face_image}
                    alt={`Person ${person.person_id + 1}`}
                    className="rounded-lg object-cover"
                    fill
                  />
                  <div className="absolute bottom-0 right-0 bg-black/50 px-2 py-1 rounded">
                    {(person.confidence_score * 100).toFixed(0)}%
                  </div>
                </div>
                <h3 className="text-lg font-semibold mb-2">Person {person.person_id}</h3>
                <p className="text-sm text-gray-400 mb-2">
                  Appearances: {person.instances_count}
                </p>
                <div className="grid grid-cols-4 gap-2">
                  {person.appearances.map((face, idx) => (
                    <div key={idx} className="relative aspect-square">
                      <Image
                        src={face.face_image}
                        alt={`Appearance ${idx + 1}`}
                        className="rounded-lg object-cover"
                        fill
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Single Instance Section */}
      {singlePeople.length > 0 && (
        <div>
          <h2 className="text-2xl font-semibold mb-4">Single Appearances ({singlePeople.length})</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {singlePeople.map((person) => (
              <div key={person.person_id} className="bg-gray-800 rounded-lg p-3">
                <div className="relative aspect-square mb-2">
                  <Image
                    src={person.representative_face.face_image}
                    alt={`Person ${person.person_id + 1}`}
                    className="rounded-lg object-cover"
                    fill
                  />
                  <div className="absolute bottom-0 right-0 bg-black/50 px-2 py-1 rounded text-xs">
                    {(person.confidence_score * 100).toFixed(0)}%
                  </div>
                </div>
                <p className="text-sm text-gray-400 truncate">
                  {person.representative_face.filename}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 