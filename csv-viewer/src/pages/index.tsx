// src/pages/index.tsx
import { useState, useEffect } from 'react';
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import Papa from 'papaparse';
import CsvTable from '../components/CsvTable';
import { CsvData } from '../types';

export default function Home() {
  const [csvData, setCsvData] = useState<CsvData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchCsvFromS3() {
      try {
        const s3Client = new S3Client({
          region: process.env.NEXT_PUBLIC_AWS_REGION,
          credentials: {
            accessKeyId: process.env.NEXT_PUBLIC_AWS_ACCESS_KEY_ID!,
            secretAccessKey: process.env.NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY!,
          },
        });

        const command = new GetObjectCommand({
          Bucket: process.env.NEXT_PUBLIC_S3_BUCKET!,
          Key: 'data.csv', // Update this path
        });

        const response = await s3Client.send(command);
        const str = await response.Body?.transformToString();
        
        if (str) {
          Papa.parse(str, {
            header: true,
            complete: (results) => {
              setCsvData(results.data as CsvData[]);
              setLoading(false);
            },
            error: (error: { message: string; }) => {
              setError('Error parsing CSV: ' + error.message);
              setLoading(false);
            }
          });
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        setError('Error fetching CSV: ' + errorMessage);
        setLoading(false);
      }
    }

    fetchCsvFromS3();
  }, []);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-red-500 text-center">
          <h2 className="text-xl font-bold mb-2">Error</h2>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-xl">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">CSV Data Viewer</h1>
      <CsvTable data={csvData} />
    </div>
  );
}